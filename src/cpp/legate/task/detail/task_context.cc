/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task_context.h>

#include <legate_defines.h>

#include <legate/data/detail/physical_store.h>
#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/macros.h>

#include <algorithm>
#include <iterator>
#include <utility>

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
#include <legate/runtime/detail/config.h>

#include <omp.h>
#else
// Still needs a definition to get the code compiled without OpenMP
// (which is safe as the call to this function is guarded by a if-constexpr)
// NOLINTNEXTLINE
#define omp_set_num_threads(n) static_cast<void>(n)
#endif

namespace legate::detail {

TaskContext::TaskContext(VariantCode variant_code,
                         Rect launch_domain,
                         Point task_index,
                         const TaskBase* task,
                         SmallVector<InternalSharedPtr<PhysicalStore>> inputs,
                         SmallVector<InternalSharedPtr<PhysicalStore>> outputs,
                         SmallVector<InternalSharedPtr<PhysicalStore>> reductions,
                         SmallVector<InternalSharedPtr<Scalar>> scalars)
  : variant_kind_{variant_code},
    launch_domain_{std::move(launch_domain)},
    task_index_{std::move(task_index)},
    op_task_{task},
    inputs_{std::move(inputs)},
    outputs_{std::move(outputs)},
    reductions_{std::move(reductions)},
    scalars_{std::move(scalars)}
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_OPENMP)) {
    if (variant_kind_ == VariantCode::OMP) {
      const auto n = Runtime::get_runtime().config().num_omp_threads();

      omp_set_num_threads(static_cast<std::int32_t>(n));
    }
  }
}

LocalTaskID TaskContext::task_id() const noexcept { return op_task_->local_task_id(); }

std::string_view TaskContext::library_name() const noexcept
{
  return op_task_->library().get_library_name().as_string_view();
}

bool TaskContext::is_single_task() const noexcept { return launch_domain_.dim() == 0; }

const Point& TaskContext::get_task_index() const noexcept { return task_index_; }

const Rect& TaskContext::get_launch_domain() const noexcept { return launch_domain_; }

std::string_view TaskContext::get_provenance() const noexcept
{
  return op_task_->provenance().as_string_view();
}

const mapping::detail::Machine& TaskContext::machine() const noexcept
{
  return op_task_->machine();
}

void TaskContext::make_all_unbound_stores_empty() {}

CUstream TaskContext::get_task_stream() const { return Runtime::get_runtime().get_cuda_stream(); }

void TaskContext::concurrent_task_barrier() {}

}  // namespace legate::detail
