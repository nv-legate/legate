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

void DataMovementRequests::register_copy_out(CopyRequest&& copy_out)
{
  copy_outs_.emplace_back(std::move(copy_out));
}

void DataMovementRequests::register_remap_copy_in(RemapRequest&& copy_in)
{
  remap_copy_ins_.emplace_back(std::move(copy_in));
}

void DataMovementRequests::register_remap_copy_out(RemapRequest&& copy_out)
{
  remap_copy_outs_.emplace_back(std::move(copy_out));
}

std::optional<InternalSharedPtr<InlineStorage>> DataMovementRequests::find_mapped_alloc(
  std::uint64_t storage_id) const
{
  const auto it = mapped_allocs_.find(storage_id);

  if (it == mapped_allocs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void DataMovementRequests::record_mapped_alloc(std::uint64_t storage_id,
                                               InternalSharedPtr<InlineStorage> alloc)
{
  mapped_allocs_[storage_id] = std::move(alloc);
}

std::optional<InternalSharedPtr<InlineStorage>> DataMovementRequests::find_working_copy(
  std::uint64_t storage_id) const
{
  const auto it = working_copies_.find(storage_id);

  if (it == working_copies_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void DataMovementRequests::record_working_copy(std::uint64_t storage_id,
                                               InternalSharedPtr<InlineStorage> alloc)
{
  working_copies_[storage_id] = std::move(alloc);
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

void DataMovementRequests::perform_remap_copy_ins(VariantCode variant_code) const
{
  for (auto&& copy_in : remap_copy_ins_) {
    copy_in.perform(variant_code);
  }
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

void DataMovementRequests::perform_copy_outs(VariantCode variant_code) const
{
  for (auto&& copy_out : copy_outs_) {
    copy_out.perform(variant_code);
  }
}

void DataMovementRequests::perform_remap_copy_outs(VariantCode variant_code) const
{
  for (auto&& copy_out : remap_copy_outs_) {
    copy_out.perform(variant_code);
  }
}

bool DataMovementRequests::empty() const
{
  return remap_copy_ins_.empty() && copy_ins_.empty() && copy_outs_.empty() &&
         remap_copy_outs_.empty() && reductions_.empty();
}

}  // namespace legate::detail
