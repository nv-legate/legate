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

namespace legate::detail {

// TODO(wonchanl): Only support 1d reductions on unpartitioned stores, should be extended to other
// cases.
class CopyRequest {
 public:
  CopyRequest(InternalSharedPtr<InlineStorage> source,
              InternalSharedPtr<InlineStorage> target,
              InternalSharedPtr<Partition> source_partition,
              InternalSharedPtr<Partition> target_partition,
              const Domain& sub_domain,
              std::uint32_t field_size);
  void perform(VariantCode variant_code) const;

  CopyRequest(const CopyRequest&)            = default;
  CopyRequest& operator=(const CopyRequest&) = default;

  CopyRequest(CopyRequest&&) noexcept            = default;
  CopyRequest& operator=(CopyRequest&&) noexcept = default;

 private:
  void cpu_all_gather_() const;
  void shuffle_(VariantCode variant_code) const;

  InternalSharedPtr<InlineStorage> source_{};
  InternalSharedPtr<InlineStorage> target_{};
  InternalSharedPtr<Partition> source_partition_{};
  InternalSharedPtr<Partition> target_partition_{};
  Domain sub_domain_{};
  std::uint32_t field_size_{};
};

}  // namespace legate::detail

#include <legate/data/detail/copy_request.inl>
