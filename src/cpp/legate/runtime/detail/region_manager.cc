/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/region_manager.h>

#include <legate/runtime/detail/field_manager.h>
#include <legate/runtime/detail/runtime.h>

#include <utility>

namespace legate::detail {

void RegionManager::ManagerEntry::destroy(Runtime& runtime, bool unordered) const
{
  runtime.destroy_region(region, unordered);
}

RegionManager::RegionManager(Legion::IndexSpace index_space) : index_space_{std::move(index_space)}
{
}

void RegionManager::destroy(bool unordered)
{
  auto&& runtime = Runtime::get_runtime();
  for (auto&& entry : entries_) {
    entry.destroy(runtime, unordered);
  }
  entries_.clear();
}

void RegionManager::push_entry_()
{
  auto&& runtime = Runtime::get_runtime();
  entries_.emplace_back(runtime.create_region(index_space_, runtime.create_field_space()));
}

bool RegionManager::has_space() const { return !entries_.empty() && active_entry_().has_space(); }

std::pair<Legion::LogicalRegion, Legion::FieldID> RegionManager::allocate_field(
  std::size_t field_size)
{
  if (!has_space()) {
    push_entry_();
  }

  auto& entry = active_entry_();
  auto fid    = Runtime::get_runtime().allocate_field(
    entry.region.get_field_space(), entry.get_next_field_id(), field_size);
  return {entry.region, fid};
}

void RegionManager::import_region(const Legion::LogicalRegion& region, std::uint32_t num_fields)
{
  entries_.emplace_back(region, num_fields);
}

}  // namespace legate::detail
