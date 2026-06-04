/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/inline_task_body.h>

#include <legate/comm/communicator.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/scalar.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/task.h>
#include <legate/operation/detail/task_store_arg.h>
#include <legate/partitioning/detail/strategy.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/task.h>
#include <legate/task/detail/task_context.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/task_context.h>
#include <legate/utilities/assert.h>
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
                    const Domain& launch_domain,
                    const DomainPoint& task_index,
                    SmallVector<InternalSharedPtr<PhysicalStore>> inputs,
                    SmallVector<InternalSharedPtr<PhysicalStore>> outputs,
                    SmallVector<InternalSharedPtr<PhysicalStore>> reductions,
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

  const Domain launch_domain_{};
  const DomainPoint task_index_{};
  const TaskBase* op_task_{};
};

// ==========================================================================================

const TaskBase& InlineTaskContext::task_() const { return *op_task_; }

// ==========================================================================================

InlineTaskContext::InlineTaskContext(VariantCode variant_code,
                                     const Domain& launch_domain,
                                     const DomainPoint& task_index,
                                     SmallVector<InternalSharedPtr<PhysicalStore>> inputs,
                                     SmallVector<InternalSharedPtr<PhysicalStore>> outputs,
                                     SmallVector<InternalSharedPtr<PhysicalStore>> reductions,
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
    launch_domain_{launch_domain},
    task_index_{task_index},
    op_task_{task}
{
}

GlobalTaskID InlineTaskContext::task_id() const noexcept
{
  const auto& task = task_();

  return task.library().get_task_id(task.local_task_id());
}

bool InlineTaskContext::is_single_task() const noexcept { return !launch_domain_.is_valid(); }

const DomainPoint& InlineTaskContext::get_task_index() const noexcept { return task_index_; }

const Domain& InlineTaskContext::get_launch_domain() const noexcept { return launch_domain_; }

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
 * @brief Extract the physical store from the task store argument
 *
 * @param elem The param to extract from
 * @param policy The instance mapping policy to use to get the right store location.
 * @param ignore_future_mutability If the store is future-backed, and has write privileges, we
 * need to ignore the normal immutability checks for futures.
 *
 * @return The physical store.
 */
[[nodiscard]] InternalSharedPtr<PhysicalStore> map_store(
  const TaskStoreArg& elem,
  const Partition* partition,
  const mapping::InstanceMappingPolicy& policy,
  const Domain& launch_domain,
  const DomainPoint& task_index)
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

  return elem.store->map(partition, launch_domain, task_index, policy);
}

[[nodiscard]] SmallVector<InternalSharedPtr<PhysicalStore>> map_stores(
  Span<const TaskStoreArg> src,
  Span<const mapping::InstanceMappingPolicy> mapping_policies,
  const Strategy& strategy,
  const Domain& launch_domain,
  const DomainPoint& task_index)
{
  SmallVector<InternalSharedPtr<PhysicalStore>> dest;

  dest.reserve(src.size());
  for (auto&& [elem, policy] : zip_equal(src, mapping_policies)) {
    dest.emplace_back(
      map_store(elem, strategy[*elem.variable].get(), policy, launch_domain, task_index));
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

[[nodiscard]] InlineTaskContext make_inline_task_context(const TaskBase& task,
                                                         VariantCode variant_code,
                                                         const Strategy& strategy)
{
  auto launch_domain    = strategy.launch_domain();
  const auto task_index = strategy.task_index();

  const auto mapping_policies = make_store_mapping_policies(task, variant_code);

  auto inputs =
    map_stores(task.inputs(), mapping_policies.input_policies, strategy, launch_domain, task_index);
  auto outputs = map_stores(
    task.outputs(), mapping_policies.output_policies, strategy, launch_domain, task_index);
  auto reductions = map_stores(
    task.reductions(), mapping_policies.reduction_policies, strategy, launch_domain, task_index);

  return InlineTaskContext{variant_code,
                           launch_domain,
                           task_index,
                           std::move(inputs),
                           std::move(outputs),
                           std::move(reductions),
                           SmallVector<InternalSharedPtr<Scalar>>{task.scalars()},
                           &task};
}

template <typename F>
void execute_task(const TaskBase& task,
                  VariantCode variant_code,
                  VariantImpl variant_impl,
                  F&& get_task_name,
                  const Strategy& strategy)
{
  auto ctx = make_inline_task_context(task, variant_code, strategy);
  std::ignore =
    task_detail::task_body(legate::TaskContext{&ctx}, variant_impl, std::forward<F>(get_task_name));
}

}  // namespace

void inline_task_body(const TaskBase& task,
                      VariantCode variant_code,
                      VariantImpl variant_impl,
                      const Strategy& strategy)
{
  const auto _ = [variant_code] {
    Runtime::get_runtime().inline_task_start(variant_code);
    return legate::make_scope_guard([&]() noexcept { Runtime::get_runtime().inline_task_end(); });
  }();
  const auto get_task_name = [&] { return task.library().get_task_name(task.local_task_id()); };
  const auto _1 =
    task_detail::make_nvtx_range(get_task_name, [&] { return task.provenance().as_string_view(); });
  static_cast<void>(_1);

  show_progress(
    strategy.task_index(), get_task_name(), variant_code, task.provenance().as_string_view());

  execute_task(task, variant_code, variant_impl, get_task_name, strategy);

  if (variant_code == VariantCode::GPU) {
    auto&& api = cuda::detail::get_cuda_driver_api();
    api->stream_synchronize(Runtime::get_runtime().get_cuda_stream());
  }
}

}  // namespace legate::detail
