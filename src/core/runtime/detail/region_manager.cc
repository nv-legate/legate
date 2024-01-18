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

#include "core/runtime/detail/region_manager.h"

#include "core/runtime/detail/field_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/hash.h"

#include <unordered_set>
#include <utility>

namespace legate::detail {

void RegionManager::ManagerEntry::destroy(Runtime* runtime, bool unordered) const
{
  runtime->destroy_region(region, unordered);
}

// Silence pass-by-value since Legion::Domain is POD, and the move ctor just does the copy
// anyways. Unfortunately there is no way to check this programatically (e.g. via a
// static_assert).
RegionManager::RegionManager(Runtime* runtime,
                             // NOLINTNEXTLINE(modernize-pass-by-value)
                             const Legion::IndexSpace& index_space)
  : runtime_{runtime}, index_space_{index_space}
{
}

void RegionManager::destroy(bool unordered)
{
  for (auto& entry : entries_) {
    entry.destroy(runtime_, unordered);
  }
  std::unordered_set<Legion::LogicalRegion> deleted;
  for (auto& entry : imported_) {
    if (deleted.find(entry.region) != deleted.end()) {
      continue;
    }
    entry.destroy(runtime_, unordered);
    deleted.insert(entry.region);
  }
  entries_.clear();
  imported_.clear();
}

void RegionManager::record_pending_match_credit_update(ConsensusMatchingFieldManager* field_mgr)
{
  pending_match_credit_updates_.push_back(field_mgr);
}

void RegionManager::update_field_manager_match_credits()
{
  for (auto* field_mgr : pending_match_credit_updates_) {
    field_mgr->calculate_match_credit();
  }
  pending_match_credit_updates_.clear();
}

void RegionManager::push_entry()
{
  entries_.emplace_back(runtime_->create_region(index_space_, runtime_->create_field_space()));
}

bool RegionManager::has_space() const { return !entries_.empty() && active_entry().has_space(); }

std::pair<Legion::LogicalRegion, Legion::FieldID> RegionManager::allocate_field(size_t field_size)
{
  if (!has_space()) {
    push_entry();
  }

  auto& entry = active_entry();
  auto fid =
    runtime_->allocate_field(entry.region.get_field_space(), entry.get_next_field_id(), field_size);
  return {entry.region, fid};
}

void RegionManager::import_region(const Legion::LogicalRegion& region)
{
  imported_.emplace_back(region);
}

}  // namespace legate::detail
