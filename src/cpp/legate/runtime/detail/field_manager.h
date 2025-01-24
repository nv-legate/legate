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

#pragma once

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/shape.h>
#include <legate/runtime/detail/consensus_match_result.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate::detail {

class FreeFieldInfo {
 public:
  FreeFieldInfo() = default;
  FreeFieldInfo(InternalSharedPtr<Shape> shape_,
                std::uint32_t field_size_,
                Legion::LogicalRegion region,
                Legion::FieldID field_id,
                InternalSharedPtr<LogicalRegionField::PhysicalState> state);

  InternalSharedPtr<Shape> shape{};
  std::uint32_t field_size{};
  Legion::LogicalRegion region{};
  Legion::FieldID field_id{};
  InternalSharedPtr<LogicalRegionField::PhysicalState> state{};
};

class FieldManager {
 public:
  virtual ~FieldManager();

  [[nodiscard]] virtual InternalSharedPtr<LogicalRegionField> allocate_field(
    InternalSharedPtr<Shape> shape, std::uint32_t field_size);
  [[nodiscard]] virtual InternalSharedPtr<LogicalRegionField> import_field(
    InternalSharedPtr<Shape> shape,
    std::uint32_t field_size,
    Legion::LogicalRegion region,
    Legion::FieldID field_id);
  virtual void free_field(FreeFieldInfo info, bool unordered);

 protected:
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> try_reuse_field_(
    const InternalSharedPtr<Shape>& shape, std::uint32_t field_size);
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> create_new_field_(
    InternalSharedPtr<Shape> shape, std::uint32_t field_size);

  using OrderedQueueKey = std::pair<Legion::IndexSpace, std::uint32_t>;
  // This is a sanitized list of (region,field_id) pairs that is guaranteed to be ordered across all
  // the shards even with control replication.
  std::unordered_map<OrderedQueueKey, std::queue<FreeFieldInfo>, hasher<OrderedQueueKey>>
    ordered_free_fields_{};
};

class MatchItem {
 public:
  MatchItem() = default;
  MatchItem(Legion::RegionTreeID tid, Legion::FieldID fid);

  Legion::RegionTreeID tid{};
  Legion::FieldID fid{};

  bool operator==(const MatchItem& rhs) const;
  [[nodiscard]] std::size_t hash() const noexcept;
};

class ConsensusMatchingFieldManager final : public FieldManager {
 public:
  ~ConsensusMatchingFieldManager() final;

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> allocate_field(
    InternalSharedPtr<Shape> shape, std::uint32_t field_size) override;
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> import_field(
    InternalSharedPtr<Shape> shape,
    std::uint32_t field_size,
    Legion::LogicalRegion region,
    Legion::FieldID field_id) override;
  void free_field(FreeFieldInfo info, bool unordered) override;

 private:
  [[nodiscard]] std::uint32_t calculate_match_credit_(const InternalSharedPtr<Shape>& shape,
                                                      std::uint32_t field_size) const;
  void maybe_issue_field_match_(const InternalSharedPtr<Shape>& shape, std::uint32_t field_size);
  void issue_field_match_();
  void process_outstanding_match_();

  std::uint32_t field_match_counter_{};
  // This list contains the fields that we know have been freed on this shard, but may not have been
  // freed yet on other shards.
  std::vector<FreeFieldInfo> unordered_free_fields_{};
  std::optional<ConsensusMatchResult<MatchItem>> outstanding_match_{};
  std::unordered_map<MatchItem, FreeFieldInfo, hasher<MatchItem>> info_for_match_items_{};
};

}  // namespace legate::detail

#include <legate/runtime/detail/field_manager.inl>
