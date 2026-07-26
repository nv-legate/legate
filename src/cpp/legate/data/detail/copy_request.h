/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>
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
  CopyRequest(std::uint64_t storage_id,
              InternalSharedPtr<Buffer> source,
              InternalSharedPtr<Buffer> target,
              InternalSharedPtr<Partition> source_partition,
              InternalSharedPtr<Partition> target_partition,
              const Rect& source_domain,
              const Rect& target_domain,
              InternalSharedPtr<Type> type);
  void perform(VariantCode variant_code) const;

  CopyRequest(const CopyRequest&)            = default;
  CopyRequest& operator=(const CopyRequest&) = default;

  CopyRequest(CopyRequest&&) noexcept            = default;
  CopyRequest& operator=(CopyRequest&&) noexcept = default;

 private:
  void all_gather_(VariantCode variant_code) const;
  void shuffle_(VariantCode variant_code) const;

  std::uint64_t storage_id_{};
  InternalSharedPtr<Buffer> source_{};
  InternalSharedPtr<Buffer> target_{};
  InternalSharedPtr<Partition> source_partition_{};
  InternalSharedPtr<Partition> target_partition_{};
  Rect source_domain_{};
  Rect target_domain_{};
  InternalSharedPtr<Type> type_{};
};

}  // namespace legate::detail

#include <legate/data/detail/copy_request.inl>
