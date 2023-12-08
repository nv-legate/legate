/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/runtime/detail/consensus_match_result.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/typedefs.h"

#include <deque>
#include <map>
#include <vector>

namespace legate::detail {

class LogicalRegionField;
class Runtime;

struct FreeFieldInfo {
  FreeFieldInfo() = default;
  FreeFieldInfo(Legion::LogicalRegion region,
                Legion::FieldID field_id,
                Legion::Future can_dealloc,
                void* attachment);

  Legion::LogicalRegion region{};
  Legion::FieldID field_id{};
  Legion::Future can_dealloc{};
  void* attachment{};
};

struct MatchItem {
  MatchItem() = default;
  MatchItem(Legion::RegionTreeID tid, Legion::FieldID fid);

  Legion::RegionTreeID tid{};
  Legion::FieldID fid{};

  friend bool operator<(const MatchItem& l, const MatchItem& r);
};

class FieldManager {
 private:
  friend LogicalRegionField;

 public:
  FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size);

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> allocate_field();
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> import_field(
    const Legion::LogicalRegion& region, Legion::FieldID field_id);

 private:
  void free_field(const Legion::LogicalRegion& region,
                  Legion::FieldID field_id,
                  Legion::Future can_dealloc,
                  void* attachment,
                  bool unordered);
  void issue_field_match();
  void process_next_field_match();

  Runtime* runtime_{};
  Domain shape_{};
  uint32_t field_size_{};
  uint32_t field_match_counter_{};
  uint32_t field_match_credit_{1};

  // This is a sanitized list of (region,field_id) pairs that is guaranteed to be ordered across all
  // the shards even with control replication.
  std::deque<FreeFieldInfo> ordered_free_fields_;
  // This list contains the fields that we know have been freed on this shard, but may not have been
  // freed yet on other shards.
  std::vector<FreeFieldInfo> unordered_free_fields_;

  std::deque<ConsensusMatchResult<MatchItem>> matches_;
  std::deque<std::map<MatchItem, FreeFieldInfo>> info_for_match_items_;
};

}  // namespace legate::detail

#include "core/runtime/detail/field_manager.inl"
