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

[[nodiscard]] SmallVector<InternalSharedPtr<PhysicalStore>> map_stores(
  Span<const TaskStoreArg> store_args,
  const legate::mapping::detail::Machine& machine,
  const Strategy& strategy,
  const Domain& launch_domain,
  const DomainPoint& task_index)
{
  SmallVector<InternalSharedPtr<PhysicalStore>> result;

  result.reserve(store_args.size());
  for (auto&& [_, store, variable] : store_args) {
    result.emplace_back(store->map(strategy[*variable].get(),
                                   strategy.find_store_projection(*variable),
                                   launch_domain,
                                   task_index,
                                   machine));
  }
  return result;
}

[[nodiscard]] TaskContext make_inline_task_context(const TaskBase& task,
                                                   VariantCode variant_code,
                                                   const Strategy& strategy)
{
  auto launch_domain    = strategy.launch_domain();
  const auto task_index = strategy.task_index();

  auto inputs  = map_stores(task.inputs(), task.machine(), strategy, launch_domain, task_index);
  auto outputs = map_stores(task.outputs(), task.machine(), strategy, launch_domain, task_index);
  auto reductions =
    map_stores(task.reductions(), task.machine(), strategy, launch_domain, task_index);

  return TaskContext{variant_code,
                     launch_domain,
                     task_index,
                     &task,
                     std::move(inputs),
                     std::move(outputs),
                     std::move(reductions),
                     SmallVector<InternalSharedPtr<Scalar>>{task.scalars()}};
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
  const auto get_task_name = [&] { return task.library().get_task_name(task.local_task_id()); };

  show_progress(
    strategy.task_index(), get_task_name(), variant_code, task.provenance().as_string_view());

  execute_task(task, variant_code, variant_impl, get_task_name, strategy);
}

}  // namespace legate::detail
