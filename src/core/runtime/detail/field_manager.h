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

#include "core/data/detail/attachment.h"
#include "core/data/detail/shape.h"
#include "core/runtime/detail/consensus_match_result.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/typedefs.h"

#include <deque>
#include <map>
#include <vector>

namespace legate::detail {

class LogicalRegionField;
class RegionManager;
class Runtime;

class FreeFieldInfo {
 public:
  FreeFieldInfo() = default;
  FreeFieldInfo(Legion::LogicalRegion region,
                Legion::FieldID field_id,
                Legion::Future can_dealloc,
                std::unique_ptr<Attachment> attachment);

  Legion::LogicalRegion region{};
  Legion::FieldID field_id{};
  Legion::Future can_dealloc{};
  std::unique_ptr<Attachment> attachment{};
};

class FieldManager {
 public:
  FieldManager(InternalSharedPtr<Shape> shape, std::uint32_t field_size);
  virtual ~FieldManager();

  [[nodiscard]] virtual InternalSharedPtr<LogicalRegionField> allocate_field();
  virtual void free_field(FreeFieldInfo free_field_info, bool unordered);

 protected:
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> try_reuse_field();
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> create_new_field();

  InternalSharedPtr<Shape> shape_{};
  std::uint32_t field_size_{};

  // This is a sanitized list of (region,field_id) pairs that is guaranteed to be ordered across all
  // the shards even with control replication.
  std::deque<FreeFieldInfo> ordered_free_fields_;
};

class MatchItem {
 public:
  MatchItem() = default;
  MatchItem(Legion::RegionTreeID tid, Legion::FieldID fid);

  Legion::RegionTreeID tid{};
  Legion::FieldID fid{};

  friend bool operator<(const MatchItem& l, const MatchItem& r);
};

class ConsensusMatchingFieldManager final : public FieldManager {
 public:
  ConsensusMatchingFieldManager(InternalSharedPtr<Shape> shape, std::uint32_t field_size);
  ~ConsensusMatchingFieldManager() final;

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> allocate_field() override;
  void free_field(FreeFieldInfo free_field_info, bool unordered) override;

  void calculate_match_credit(const Shape* initiator);

 private:
  void issue_field_match();
  void process_next_field_match();

  std::uint32_t field_match_counter_{};
  std::uint32_t field_match_credit_{1};

  // This list contains the fields that we know have been freed on this shard, but may not have been
  // freed yet on other shards.
  std::vector<FreeFieldInfo> unordered_free_fields_;

  std::deque<ConsensusMatchResult<MatchItem>> matches_;
  std::deque<std::map<MatchItem, FreeFieldInfo>> info_for_match_items_;
};

}  // namespace legate::detail

#include "core/runtime/detail/field_manager.inl"
