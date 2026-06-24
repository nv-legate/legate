/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/data_movement_requests.h>

namespace legate::detail {

void DataMovementRequests::register_copy_in(CopyRequest&& copy_in)
{
  copy_ins_.emplace_back(std::move(copy_in));
}

void DataMovementRequests::register_reduction(ReductionRequest&& reduction)
{
  reductions_.emplace_back(std::move(reduction));
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
