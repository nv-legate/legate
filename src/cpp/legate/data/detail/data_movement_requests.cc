/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/data_movement_requests.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void DataMovementRequests::register_copy_in(CopyRequest&& copy_in)
{
  copy_ins_.emplace_back(std::move(copy_in));
}

void DataMovementRequests::register_reduction(ReductionRequest&& reduction)
{
  reductions_.emplace_back(std::move(reduction));
}

void DataMovementRequests::perform_data_sync() const
{
  if (!data_sync_required_) {
    return;
  }

  auto&& runtime = Runtime::get_runtime();
  auto&& api     = cuda::detail::get_cuda_driver_api();

  LEGATE_ASSERT(runtime.config().need_cuda());
  // TODO(wonchanl): This synchronization shouldn't happen in the sequencing thread, but should be
  // performed by the thread running CPU "kernels".
  api->stream_synchronize(runtime.get_cuda_stream());
}

void DataMovementRequests::perform_copy_ins(VariantCode variant_code) const
{
  for (auto&& copy_in : copy_ins_) {
    copy_in.perform(variant_code);
  }
}

void DataMovementRequests::perform_reductions(VariantCode variant_code) const
{
  for (auto&& reduction : reductions_) {
    reduction.perform(variant_code);
  }
}

bool DataMovementRequests::empty() const { return copy_ins_.empty() && reductions_.empty(); }

}  // namespace legate::detail
