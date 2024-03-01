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

#include "core/utilities/typedefs.h"

#include <utility>
#include <vector>

namespace legate::detail {

class ConsensusMatchingFieldManager;
class Runtime;
class Shape;

class RegionManager {
 public:
  static constexpr Legion::FieldID FIELD_ID_BASE = 10000;
  static constexpr std::uint32_t MAX_NUM_FIELDS  = LEGION_MAX_FIELDS - LEGION_DEFAULT_LOCAL_FIELDS;

 private:
  class ManagerEntry {
   public:
    explicit ManagerEntry(const Legion::LogicalRegion& _region) : region{_region} {}
    ManagerEntry(const Legion::LogicalRegion& _region, std::uint32_t num_fields)
      : region{_region}, next_field_id{FIELD_ID_BASE + num_fields}
    {
    }

    [[nodiscard]] bool has_space() const { return next_field_id - FIELD_ID_BASE < MAX_NUM_FIELDS; }
    [[nodiscard]] Legion::FieldID get_next_field_id() { return next_field_id++; }

    void destroy(Runtime* runtime, bool unordered) const;

    Legion::LogicalRegion region{};
    Legion::FieldID next_field_id{FIELD_ID_BASE};
  };

 public:
  explicit RegionManager(Legion::IndexSpace index_space);
  void destroy(bool unordered = false);
  void record_pending_match_credit_update(ConsensusMatchingFieldManager* field_mgr);
  void update_field_manager_match_credits(const Shape* initiator);

 private:
  [[nodiscard]] const ManagerEntry& active_entry() const { return entries_.back(); }
  [[nodiscard]] ManagerEntry& active_entry() { return entries_.back(); }
  void push_entry();

 public:
  [[nodiscard]] bool has_space() const;
  [[nodiscard]] std::pair<Legion::LogicalRegion, Legion::FieldID> allocate_field(
    std::size_t field_size);
  void import_region(const Legion::LogicalRegion& region, std::uint32_t num_fields);

 private:
  Legion::IndexSpace index_space_{};
  std::vector<ManagerEntry> entries_{};
  std::vector<ConsensusMatchingFieldManager*> pending_match_credit_updates_{};
};

}  // namespace legate::detail
