/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/extract_scalar.h>

#include <legate/task/detail/legion_task_body.h>
#include <legate/task/detail/task.h>
#include <legate/task/task_context.h>
#include <legate/utilities/machine.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace legate::detail {

namespace {

[[nodiscard]] Legion::UntypedDeferredValue task_impl(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>& regions,
  Legion::Context legion_context,
  Legion::Runtime* runtime,
  VariantCode code)
{
  show_progress(task, legion_context, runtime);

  const auto& future       = task->futures[0];
  auto legion_task_context = LegionTaskContext{*task, code, regions};
  const auto context       = legate::TaskContext{&legion_task_context};
  const auto offset        = context.scalar(0).value<std::size_t>();
  const auto size          = [&] {
    const auto in_size = context.scalar(1).value<std::size_t>();

    return std::min(in_size, future.get_untyped_size() - offset);
  }();

  const auto mem_kind   = find_memory_kind_for_executing_processor();
  const auto* const ptr = static_cast<const std::byte*>(future.get_buffer(mem_kind)) + offset;

  auto return_value = Legion::UntypedDeferredValue{size, mem_kind};
  const auto acc    = AccessorWO<std::byte, 1>{return_value, size, false};

  std::memcpy(acc.ptr(0), ptr, size);
  return return_value;
}

}  // namespace

/* static */ Legion::UntypedDeferredValue ExtractScalar::cpu_variant(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>& regions,
  Legion::Context context,
  Legion::Runtime* runtime)
{
  return task_impl(task, regions, context, runtime, VariantCode::CPU);
}

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
/* static */ Legion::UntypedDeferredValue ExtractScalar::omp_variant(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>& regions,
  Legion::Context context,
  Legion::Runtime* runtime)
{
  return task_impl(task, regions, context, runtime, VariantCode::OMP);
}
#endif

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
/* static */ Legion::UntypedDeferredValue ExtractScalar::gpu_variant(
  const Legion::Task* task,
  const std::vector<Legion::PhysicalRegion>& regions,
  Legion::Context context,
  Legion::Runtime* runtime)
{
  return task_impl(task, regions, context, runtime, VariantCode::GPU);
}
#endif

}  // namespace legate::detail
