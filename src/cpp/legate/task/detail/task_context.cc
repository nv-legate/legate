/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task_context.h>

#include <legate_defines.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/physical_stores/unbound_physical_store.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/macros.h>

#include <algorithm>
#include <iterator>

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

TaskContext::TaskContext(CtorArgs&& args)
  : variant_kind_{args.variant_kind},
    inputs_{std::move(args.inputs)},
    outputs_{std::move(args.outputs)},
    reductions_{std::move(args.reductions)},
    scalars_{std::move(args.scalars)},
    comms_{std::move(args.comms)},
    can_raise_exception_{args.can_raise_exception},
    can_elide_device_ctx_sync_{args.can_elide_device_ctx_sync}
{
  // Make copies of stores that we need to postprocess, as clients might move the stores away.
  for (auto&& store : outputs_) {
    if (dynamic_cast<const UnboundPhysicalStore*>(store.get())) {
      unbound_stores_.push_back(store);
    } else if (dynamic_cast<const FuturePhysicalStore*>(store.get())) {
      scalar_stores_.push_back(store);
    }
  }

  std::copy_if(reductions_.begin(),
               reductions_.end(),
               std::back_inserter(scalar_stores_),
               [](const InternalSharedPtr<PhysicalStore>& store) -> bool {
                 return dynamic_cast<const FuturePhysicalStore*>(store.get());
               });

  if constexpr (LEGATE_DEFINED(LEGATE_USE_OPENMP)) {
    if (variant_kind_ == VariantCode::OMP) {
      const auto n = Runtime::get_runtime().config().num_omp_threads();

      omp_set_num_threads(static_cast<std::int32_t>(n));
    }
  }
}

void TaskContext::make_all_unbound_stores_empty()
{
  for (auto&& store : get_unbound_stores_()) {
    store->as_unbound_store().bind_empty_data();
  }
}

CUstream TaskContext::get_task_stream() const { return Runtime::get_runtime().get_cuda_stream(); }

void TaskContext::concurrent_task_barrier()
{
  // check whether we run rank per GPU or rank per node
  if (!Runtime::get_runtime().is_rank_per_gpu()) {
    Runtime::get_runtime().get_legion_runtime()->concurrent_task_barrier(
      Legion::Runtime::get_context());
  }
}

}  // namespace legate::detail
