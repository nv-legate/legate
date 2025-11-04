/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/inline_task_body.h>

#include <legate/comm/communicator.h>
#include <legate/data/detail/physical_array.h>
#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/scalar.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/task.h>
#include <legate/operation/detail/task_array_arg.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/task.h>
#include <legate/task/detail/task_context.h>
#include <legate/task/task_context.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/store_iterator_cache.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/scope_guard.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string_view>
#include <utility>

namespace legate::detail {

namespace {

class InlineTaskContext final : public TaskContext {
 public:
  InlineTaskContext(VariantCode variant_code,
                    SmallVector<InternalSharedPtr<PhysicalArray>> inputs,
                    SmallVector<InternalSharedPtr<PhysicalArray>> outputs,
                    SmallVector<InternalSharedPtr<PhysicalArray>> reductions,
                    SmallVector<InternalSharedPtr<Scalar>> scalars,
                    const TaskBase* task);

  [[nodiscard]] GlobalTaskID task_id() const noexcept override;
  [[nodiscard]] bool is_single_task() const noexcept override;
  [[nodiscard]] const DomainPoint& get_task_index() const noexcept override;
  [[nodiscard]] const Domain& get_launch_domain() const noexcept override;
  [[nodiscard]] std::string_view get_provenance() const noexcept override;
  [[nodiscard]] const mapping::detail::Machine& machine() const noexcept override;

 private:
  [[nodiscard]] const TaskBase& task_() const;

  const TaskBase* op_task_{};
};

// ==========================================================================================

const TaskBase& InlineTaskContext::task_() const { return *op_task_; }

// ==========================================================================================

InlineTaskContext::InlineTaskContext(VariantCode variant_code,
                                     SmallVector<InternalSharedPtr<PhysicalArray>> inputs,
                                     SmallVector<InternalSharedPtr<PhysicalArray>> outputs,
                                     SmallVector<InternalSharedPtr<PhysicalArray>> reductions,
                                     SmallVector<InternalSharedPtr<Scalar>> scalars,
                                     const TaskBase* task)
  : TaskContext{{variant_code,
                 task->can_throw_exception(),
                 task->can_elide_device_ctx_sync(),
                 std::move(inputs),
                 std::move(outputs),
                 std::move(reductions),
                 std::move(scalars),
                 SmallVector<legate::comm::Communicator>{}}},
    op_task_{task}
{
}

GlobalTaskID InlineTaskContext::task_id() const noexcept
{
  const auto& task = task_();

  return task.library().get_task_id(task.local_task_id());
}

bool InlineTaskContext::is_single_task() const noexcept { return true; }

const DomainPoint& InlineTaskContext::get_task_index() const noexcept
{
  static const DomainPoint p{};

  return p;
}

const Domain& InlineTaskContext::get_launch_domain() const noexcept
{
  static const auto launch_domain = Domain{DomainPoint{0}, DomainPoint{1}};

  return launch_domain;
}

std::string_view InlineTaskContext::get_provenance() const noexcept
{
  return task_().provenance().as_string_view();
}

const mapping::detail::Machine& InlineTaskContext::machine() const noexcept
{
  return task_().machine();
}

// ==========================================================================================

[[nodiscard]] SmallVector<InternalSharedPtr<PhysicalArray>> fill_vector(
  Span<const TaskArrayArg> src,
  bool ignore_future_mutability,
  const StoreIteratorCache<InternalSharedPtr<PhysicalStore>>& get_stores_cache,
  SmallVector<Legion::UntypedDeferredValue>* deferred_buffers)
{
  SmallVector<InternalSharedPtr<PhysicalArray>> dest;

  dest.reserve(src.size());
  for (auto&& elem : src) {
    const auto phys_array = std::visit(
      Overload{
        [&](const InternalSharedPtr<LogicalArray>& arr) -> InternalSharedPtr<PhysicalArray> {
          // For LogicalArray (AutoTask/ManualTask), convert to PhysicalArray
          return arr->get_physical_array(legate::mapping::StoreTarget::SYSMEM,
                                         ignore_future_mutability);
        },
        [&](const InternalSharedPtr<PhysicalArray>& arr) -> InternalSharedPtr<PhysicalArray> {
          // For PhysicalArray (PhysicalTask), use directly
          return arr;
        }},
      elem.array);

    dest.emplace_back(phys_array);

    if (!ignore_future_mutability) {
      continue;
    }

    for (auto&& phys_store : get_stores_cache(*phys_array)) {
      if (phys_store->kind() == PhysicalStore::Kind::FUTURE) {
        // Was an output scalar or scalar reduction, save a reference to the deferred buffer so
        // we can create a new future out of it down the line in finalize().
        deferred_buffers->emplace_back(phys_store->as_future_store().get_buffer());
      }
    }
  }
  return dest;
}

[[nodiscard]] InlineTaskContext make_inline_task_context(
  const TaskBase& task,
  VariantCode variant_code,
  SmallVector<Legion::UntypedDeferredValue>* deferred_buffers)
{
  const auto get_stores_cache = StoreIteratorCache<InternalSharedPtr<PhysicalStore>>{};

  auto inputs = fill_vector(task.inputs(), false, get_stores_cache, deferred_buffers);
  // None of the inputs should ever create an output buffer
  LEGATE_CHECK(deferred_buffers->empty());
  // We do these here instead of inline in the function arguments because the order in which
  // these are executed matters. We want to fill up the deferred_bufs_ vector first with output
  // scalars, then with reduction scalars.
  //
  // This is based on the assumption that:
  //
  // for log_arr in task.outputs():
  //   for phy_st in log_arr.get_physical_array().stores():
  //     if phy_st.is_scalar():
  //       yield phy_st
  //
  // and
  //
  // for log_st in task.scalar_outputs():
  //   phy_st = log_st.get_physical_store()
  //   if phy_st.is_scalar():
  //     yield phy_st
  //
  // return stores in the same order (and similarly for reductions). If this is no longer the
  // case, then the set of "scalar_set_future()" loops in the task body will no longer be
  // correct.

  // Reserve space for scalar outputs and reductions based on task type
  if (const auto* logical_task = dynamic_cast<const LogicalTask*>(&task)) {
    deferred_buffers->reserve(logical_task->scalar_outputs().size() +
                              logical_task->scalar_reductions().size());
  } else if (const auto* physical_task = dynamic_cast<const PhysicalTask*>(&task)) {
    deferred_buffers->reserve(physical_task->physical_scalar_outputs().size() +
                              physical_task->physical_scalar_reductions().size());
  }
  auto outputs    = fill_vector(task.outputs(), true, get_stores_cache, deferred_buffers);
  auto reductions = fill_vector(task.reductions(), true, get_stores_cache, deferred_buffers);

  return InlineTaskContext{variant_code,
                           std::move(inputs),
                           std::move(outputs),
                           std::move(reductions),
                           SmallVector<InternalSharedPtr<Scalar>>{task.scalars()},
                           &task};
}

template <typename F>
[[nodiscard]] std::pair<std::optional<ReturnedException>, SmallVector<Legion::UntypedDeferredValue>>
execute_task(const TaskBase& task,
             VariantCode variant_code,
             VariantImpl variant_impl,
             F&& get_task_name)
{
  auto deferred_buffers = SmallVector<Legion::UntypedDeferredValue>{};
  auto ctx              = make_inline_task_context(task, variant_code, &deferred_buffers);
  auto exn =
    task_detail::task_body(legate::TaskContext{&ctx}, variant_impl, std::forward<F>(get_task_name));

  return {std::move(exn), std::move(deferred_buffers)};
}

// Function overloads for different task types (replaces template specializations)
void handle_return_values_impl(const LogicalTask& task,
                               Span<const Legion::UntypedDeferredValue> deferred_buffers);
void handle_return_values_impl(const PhysicalTask& task,
                               Span<const Legion::UntypedDeferredValue> deferred_buffers);

// Implementation for LogicalTask (AutoTask/ManualTask behavior)
void handle_return_values_impl(const LogicalTask& task,
                               Span<const Legion::UntypedDeferredValue> deferred_buffers)
{
  // Order of deferred_buffers and scalar_outputs, scalar_reductions must be the same. See
  // make_inline_task_context() for more details.
  const auto scalar_set_future = [&](const InternalSharedPtr<LogicalStore>& scal,
                                     const Legion::UntypedDeferredValue& buf) {
    const auto size = scal->storage_size();
    auto* ptr       = AccessorRO<std::int8_t, 1>{buf, size}.ptr(0);
    auto fut        = Legion::Future::from_untyped_pointer(
      ptr,
      size,
      // Don't take ownership because Legion already "owns" the
      // deferred value, we wouldn't want to double-free it.
      false,
      // task.provenance() is a ZStringView
      task.provenance().data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
    );
    static_assert(std::is_same_v<std::decay_t<decltype(task.provenance())>, ZStringView>);

    scal->set_future(std::move(fut));
  };
  std::size_t idx = 0;

  LEGATE_ASSERT(deferred_buffers.size() ==
                task.scalar_outputs().size() + task.scalar_reductions().size());
  for (auto&& scal : task.scalar_outputs()) {
    scalar_set_future(scal, deferred_buffers[idx++]);
  }
  for (auto&& [scal, _] : task.scalar_reductions()) {
    scalar_set_future(scal, deferred_buffers[idx++]);
  }
}

// Implementation for PhysicalTask (PhysicalStore behavior)
void handle_return_values_impl(const PhysicalTask& task,
                               Span<const Legion::UntypedDeferredValue> deferred_buffers)
{
  // Direct access to PhysicalTask - no casting needed

  const auto scalar_set_future = [&](const InternalSharedPtr<PhysicalStore>& scal,
                                     const Legion::UntypedDeferredValue& buf) {
    // Create a future from the deferred buffer and update the PhysicalStore
    const auto size = scal->type()->size();  // PhysicalStore uses type()->size()
    auto* ptr       = AccessorRO<std::int8_t, 1>{buf, size}.ptr(0);
    auto fut        = Legion::Future::from_untyped_pointer(
      ptr,
      size,
      // Don't take ownership because Legion already "owns" the
      // deferred value, we wouldn't want to double-free it.
      false,
      // task.provenance() is a ZStringView
      task.provenance().data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
    );

    // Update the PhysicalStore with the new future
    scal->as_future_store().set_future(std::move(fut));
  };
  std::size_t idx = 0;

  // Access PhysicalTask's scalar data directly
  const auto& scalar_outputs    = task.physical_scalar_outputs();
  const auto& scalar_reductions = task.physical_scalar_reductions();

  LEGATE_ASSERT(deferred_buffers.size() == scalar_outputs.size() + scalar_reductions.size());

  for (auto&& scal : scalar_outputs) {
    scalar_set_future(scal, deferred_buffers[idx++]);
  }
  for (auto&& [scal, _] : scalar_reductions) {
    scalar_set_future(scal, deferred_buffers[idx++]);
  }
}

// Dispatcher function - determines which overload to use
void handle_return_values(const TaskBase& task,
                          Span<const Legion::UntypedDeferredValue> deferred_buffers)
{
  // Runtime type check to dispatch to correct overload
  if (const auto* physical_task = dynamic_cast<const PhysicalTask*>(&task)) {
    handle_return_values_impl(*physical_task, deferred_buffers);
  } else if (const auto* logical_task = dynamic_cast<const LogicalTask*>(&task)) {
    handle_return_values_impl(*logical_task, deferred_buffers);
  } else {
    LEGATE_ABORT("Unknown task type in handle_return_values");
  }
}

}  // namespace

void inline_task_body(const TaskBase& task, VariantCode variant_code, VariantImpl variant_impl)
{
  const auto _ = [] {
    Runtime::get_runtime().inline_task_start();
    return legate::make_scope_guard([&]() noexcept { Runtime::get_runtime().inline_task_end(); });
  }();
  const auto get_task_name = [&] { return task.library().get_task_name(task.local_task_id()); };
  const auto _1 =
    task_detail::make_nvtx_range(get_task_name, [&] { return task.provenance().as_string_view(); });
  static_cast<void>(_1);

  show_progress({}, get_task_name(), task.provenance().as_string_view());

  auto [exception, deferred_buffers] =
    execute_task(task, variant_code, variant_impl, get_task_name);

  handle_return_values(task, deferred_buffers);

  if (exception.has_value()) {
    detail::Runtime::get_runtime().record_pending_exception(*std::move(exception));
  }
}

}  // namespace legate::detail
