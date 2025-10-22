/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  /**
   * @brief Issue a consesus match on discarded fields in multi-rank runs.
   */
  virtual void issue_field_match();

 protected:
  [[nodiscard]] std::optional<InternalSharedPtr<LogicalRegionField>> try_reuse_field_(
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
  /**
   * @brief Issue a consesus match on discarded fields in multi-rank runs.
   *
   * Due to the non-deterministic nature of garbage collection in managed language like
   * Python, in a multi-process run different top-level processes (Python interpreters)
   * may delete the same Legate Store at different points in time. Legate tries to
   * reuse the backing RegionFields of deleted Stores, but to do this safely all
   * instances of the Legate runtime must agree on the set of RegionFields that have been freed.
   *
   * This function issues an asynchronous collective operation, whereby all processes
   * will agree on the set of RegionFields that have been freed across all of them, so
   * they can be safely reused.
   */
  void issue_field_match() override;

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
