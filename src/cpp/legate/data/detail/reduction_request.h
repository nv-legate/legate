/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/type/detail/types.h>
#include <legate/type/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>

namespace legate::detail {

void fill_with_reduction_identity(void* dst,
                                  std::size_t count,
                                  Type::Code type_code,
                                  ReductionOpKind op_kind);

// TODO(wonchanl): Only support 1d reductions on unpartitioned stores, should be extended to other
// cases.
class ReductionRequest {
 public:
  ReductionRequest(InternalSharedPtr<InlineStorage> source,
                   InternalSharedPtr<InlineStorage> target,
                   InternalSharedPtr<Partition> source_partition,
                   InternalSharedPtr<Partition> target_partition,
                   const Domain& sub_domain,
                   InternalSharedPtr<Type> type,
                   ReductionOpKind op_kind);
  void perform(VariantCode variant_code) const;

  ReductionRequest(const ReductionRequest&)            = default;
  ReductionRequest& operator=(const ReductionRequest&) = default;

  ReductionRequest(ReductionRequest&&) noexcept            = default;
  ReductionRequest& operator=(ReductionRequest&&) noexcept = default;

 private:
  void all_reduce_(VariantCode variant_code) const;
  void reduce_to_replicated_target_(VariantCode variant_code) const;

  InternalSharedPtr<InlineStorage> source_{};
  InternalSharedPtr<InlineStorage> target_{};
  // Partition on the target (lhs) storage produced by inverting the task-launch partition
  // through the store transform stack. Drives the dispatch in `perform()`.
  InternalSharedPtr<Partition> source_partition_{};
  InternalSharedPtr<Partition> target_partition_{};
  Domain sub_domain_{};
  InternalSharedPtr<Type> type_{};
  ReductionOpKind op_kind_{};
};

}  // namespace legate::detail

#include <legate/data/detail/reduction_request.inl>
