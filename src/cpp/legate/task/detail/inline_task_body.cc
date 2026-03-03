/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/inline_task_body.h>

#include <legate/comm/communicator.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/data/detail/logical_array.h>
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
#include <legate/task/detail/task_info.h>
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
  static const auto launch_domain = Domain{DomainPoint{0}, DomainPoint{0}};

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

/**
 * @brief Extract the physical array from the task array argument
 *
 * @param elem The param to extract from
 * @param policy The instance mapping policy to use to get the right store location.
 * @param ignore_future_mutability If the array is future-backed, and has write privileges, we
 * need to ignore the normal immutability checks for futures.
 *
 * @return The physical array.
 */
[[nodiscard]] InternalSharedPtr<PhysicalArray> extract_physical_array(
  const TaskArrayArg& elem,
  const mapping::InstanceMappingPolicy& policy,
  bool ignore_future_mutability)
{
  if (policy.exact) {
    // See discussion in https://github.com/nv-legate/legate.internal/issues/3539 for why this
    // isn't supported right now
    throw TracedException<std::runtime_error>{
      "InstanceMappingPolicy::exact not yet supported in inline execution."};
  }

  if (policy.ordering.has_value() && policy.ordering->kind() != mapping::DimOrdering::Kind::C) {
    // See discussion in https://github.com/nv-legate/legate.internal/issues/3539 for why this
    // isn't supported right now
    throw TracedException<std::runtime_error>{
      "Dimension orderings other than C (or unspecified) not yet supported in inline execution."};
  }

  return std::visit(Overload{[&](const InternalSharedPtr<LogicalArray>& arr) {
                               return arr->get_physical_array(policy.target,
                                                              ignore_future_mutability);
                             },
                             [&](const InternalSharedPtr<PhysicalArray>& arr) {
                               LEGATE_CHECK(arr->data()->target() == policy.target);
                               return arr;
                             }},
                    elem.array);
}

[[nodiscard]] SmallVector<InternalSharedPtr<PhysicalArray>> fill_vector(
  Span<const TaskArrayArg> src,
  Span<const mapping::InstanceMappingPolicy> mapping_policies,
  bool ignore_future_mutability,
  const StoreIteratorCache<InternalSharedPtr<PhysicalStore>>& get_stores_cache,
  SmallVector<Legion::UntypedDeferredValue>* deferred_buffers)
{
  SmallVector<InternalSharedPtr<PhysicalArray>> dest;

  dest.reserve(src.size());
  for (auto&& [elem, policy] : zip_equal(src, mapping_policies)) {
    auto&& phys_array =
      dest.emplace_back(extract_physical_array(elem, policy, ignore_future_mutability));

    if (!ignore_future_mutability) {
      continue;
    }

    for (auto&& phys_store : get_stores_cache(*phys_array)) {
      if (const auto* const fut_store =
            dynamic_cast<const FuturePhysicalStore*>(phys_store.get())) {
        // Was an output scalar or scalar reduction, save a reference to the deferred buffer so
        // we can create a new future out of it down the line in finalize().
        deferred_buffers->emplace_back(fut_store->get_buffer());
      }
    }
  }
  return dest;
}

struct TaskStoreMappingPolicies {
  SmallVector<mapping::InstanceMappingPolicy> input_policies{};
  SmallVector<mapping::InstanceMappingPolicy> output_policies{};
  SmallVector<mapping::InstanceMappingPolicy> reduction_policies{};
};

/**
 * @brief Get the default store target options for each variant.
 *
 * @param variant_code The variant to get the options for.
 *
 * @return The default options.
 */
[[nodiscard]] Span<const mapping::StoreTarget> get_default_target_options(VariantCode variant_code)
{
  switch (variant_code) {
    case VariantCode::CPU: {
      static constexpr auto opts = std::array{mapping::StoreTarget::SYSMEM};

      return opts;
    }
    case VariantCode::GPU: {
      static constexpr auto opts =
        std::array{mapping::StoreTarget::FBMEM, mapping::StoreTarget::ZCMEM};

      return opts;
    }
    case VariantCode::OMP: {
      static constexpr auto opts =
        std::array{mapping::StoreTarget::SOCKETMEM, mapping::StoreTarget::SYSMEM};

      return opts;
    }
  }
  LEGATE_ABORT("Unhandled variant code: ", to_underlying(variant_code));
}

/**
 * @brief Create the store mapping policies for input, output, and reduction arguments for a task.
 *
 * If the user does not fully cover every store argument with their mapping policies, then any
 * unspecified arguments will receive a default mapping policy.
 *
 * @param task The task to generate the policies for.
 * @param variant_code The variant.
 *
 * @return The store mapping policies
 */
[[nodiscard]] TaskStoreMappingPolicies make_store_mapping_policies(const TaskBase& task,
                                                                   VariantCode variant_code)
{
  const auto target_options = get_default_target_options(variant_code);
  auto ret                  = [&] {
    const auto default_policy =
      mapping::InstanceMappingPolicy{}.with_target(target_options.front());

    return TaskStoreMappingPolicies{
      /* input_policies */ {tags::size_tag, task.inputs().size(), default_policy},
      /* output_policies */ {tags::size_tag, task.outputs().size(), default_policy},
      /* reduction_policies */ {tags::size_tag, task.reductions().size(), default_policy},
    };
  }();

  if (auto&& sm = task.library().find_task(task.local_task_id())->task_config()->store_mappings();
      sm.has_value()) {
    sm->apply_inline(
      task, target_options, &ret.input_policies, &ret.output_policies, &ret.reduction_policies);
  }
  return ret;
}

[[nodiscard]] std::pair<InlineTaskContext, SmallVector<Legion::UntypedDeferredValue>>
make_inline_task_context(const TaskBase& task, VariantCode variant_code)
{
  const auto get_stores_cache = StoreIteratorCache<InternalSharedPtr<PhysicalStore>>{};
  const auto mapping_policies = make_store_mapping_policies(task, variant_code);
  auto deferred_buffers       = SmallVector<Legion::UntypedDeferredValue>{};

  auto inputs = fill_vector(task.inputs(),
                            mapping_policies.input_policies,
                            /* ignore_future_mutability */ false,
                            get_stores_cache,
                            &deferred_buffers);
  // None of the inputs should ever create an output buffer
  LEGATE_CHECK(deferred_buffers.empty());
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
    deferred_buffers.reserve(logical_task->scalar_outputs().size() +
                             logical_task->scalar_reductions().size());
  } else if (const auto* physical_task = dynamic_cast<const PhysicalTask*>(&task)) {
    deferred_buffers.reserve(physical_task->physical_scalar_outputs().size() +
                             physical_task->physical_scalar_reductions().size());
  } else {
    LEGATE_ABORT("Unhandled task kind");
  }

  auto outputs    = fill_vector(task.outputs(),
                             mapping_policies.output_policies,
                             /* ignore_future_mutability */ true,
                             get_stores_cache,
                             &deferred_buffers);
  auto reductions = fill_vector(task.reductions(),
                                mapping_policies.reduction_policies,
                                /* ignore_future_mutability */ true,
                                get_stores_cache,
                                &deferred_buffers);

  return {InlineTaskContext{variant_code,
                            std::move(inputs),
                            std::move(outputs),
                            std::move(reductions),
                            SmallVector<InternalSharedPtr<Scalar>>{task.scalars()},
                            &task},
          std::move(deferred_buffers)};
}

template <typename F>
[[nodiscard]] std::pair<std::optional<ReturnedException>, SmallVector<Legion::UntypedDeferredValue>>
execute_task(const TaskBase& task,
             VariantCode variant_code,
             VariantImpl variant_impl,
             F&& get_task_name)
{
  auto [ctx, deferred_buffers] = make_inline_task_context(task, variant_code);
  auto exn =
    task_detail::task_body(legate::TaskContext{&ctx}, variant_impl, std::forward<F>(get_task_name));

  // Normally implicit device synchronization is handled for us by Realm, but in the fast-path
  // we bypass Legion/Realm entirely so we need to handle the sync ourselves.
  if (variant_code == VariantCode::GPU && !task.can_elide_device_ctx_sync() &&
      Runtime::get_runtime().config().enable_inline_task_launch()) {
    cuda::detail::sync_current_ctx();
  }

  return {std::move(exn), std::move(deferred_buffers)};
}

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

void handle_return_values_impl(const PhysicalTask& task,
                               Span<const Legion::UntypedDeferredValue> deferred_buffers)
{
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
  const auto _ = [variant_code] {
    Runtime::get_runtime().inline_task_start(variant_code);
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
