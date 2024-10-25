/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/task/detail/task_context.h"

#include "legate_defines.h"

#include "legate/cuda/cuda.h"
#include "legate/data/detail/physical_store.h"
#include "legate/runtime/detail/runtime.h"
#include "legate/utilities/detail/store_iterator_cache.h"
#include "legate/utilities/macros.h"

#include <algorithm>
#include <iterator>

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
#include "legate/runtime/detail/config.h"

#include <omp.h>
#else
// Still needs a definition to get the code compiled without OpenMP
// (which is safe as the call to this function is guarded by a if-constexpr)
// NOLINTNEXTLINE
#define omp_set_num_threads(...) \
  do {                           \
  } while (0)
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
  auto get_stores = StoreIteratorCache<InternalSharedPtr<PhysicalStore>>{};

  // Make copies of stores that we need to postprocess, as clients might move the stores away.
  for (auto&& output : outputs_) {
    for (auto&& store : get_stores(*output)) {
      if (store->is_unbound_store()) {
        unbound_stores_.push_back(std::move(store));
      } else if (store->is_future()) {
        scalar_stores_.push_back(std::move(store));
      }
    }
  }

  for (auto&& reduction : reductions_) {
    auto&& stores = get_stores(*reduction);

    std::copy_if(std::make_move_iterator(stores.begin()),
                 std::make_move_iterator(stores.end()),
                 std::back_inserter(scalar_stores_),
                 [](const InternalSharedPtr<PhysicalStore>& store) { return store->is_future(); });
  }

  if constexpr (LEGATE_DEFINED(LEGATE_USE_OPENMP)) {
    if (variant_kind_ == VariantCode::OMP) {
      omp_set_num_threads(static_cast<std::int32_t>(Config::num_omp_threads));
    }
  }
}

void TaskContext::make_all_unbound_stores_empty()
{
  for (auto&& store : get_unbound_stores_()) {
    store->bind_empty_data();
  }
}

CUstream_st* TaskContext::get_task_stream() const
{
  return Runtime::get_runtime()->get_cuda_stream();
}

void TaskContext::concurrent_task_barrier()
{
  Runtime::get_runtime()->get_legion_runtime()->concurrent_task_barrier(
    Legion::Runtime::get_context());
}

}  // namespace legate::detail
