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

#include <memory>

#include "core/utilities/typedefs.h"

namespace legate::detail {

class LogicalRegionField;
class Runtime;
template <typename T>
class ConsensusMatchResult;

class FieldManager {
 private:
  friend LogicalRegionField;

 public:
  FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size);

 public:
  std::shared_ptr<LogicalRegionField> allocate_field();
  std::shared_ptr<LogicalRegionField> import_field(const Legion::LogicalRegion& region,
                                                   Legion::FieldID field_id);

 private:
  void free_field(const Legion::LogicalRegion& region, Legion::FieldID field_id, bool unordered);
  void issue_field_match();
  void process_next_field_match();

 private:
  Runtime* runtime_;
  Domain shape_;
  uint32_t field_size_;
  uint32_t field_match_counter_{0};

 private:
  using FreeField = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  // This is a sanitized list of (region,field_id) pairs that is guaranteed to be ordered across all
  // the shards even with control replication.
  std::deque<FreeField> ordered_free_fields_;
  // This list contains the fields that we know have been freed on this shard, but may not have been
  // freed yet on other shards.
  std::vector<FreeField> unordered_free_fields_;

 private:
  struct MatchItem {
    Legion::RegionTreeID tid;
    Legion::FieldID fid;
    friend bool operator<(const MatchItem& l, const MatchItem& r)
    {
      return std::tie(l.tid, l.fid) < std::tie(r.tid, r.fid);
    }
  };
  std::deque<ConsensusMatchResult<MatchItem>> matches_;
  std::deque<std::map<MatchItem, Legion::LogicalRegion>> regions_for_match_items_;
};

}  // namespace legate::detail
