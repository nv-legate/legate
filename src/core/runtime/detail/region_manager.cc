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
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

void RegionManager::ManagerEntry::destroy(Runtime* runtime, bool unordered) const
{
  runtime->destroy_region(region, unordered);
}

RegionManager::RegionManager(Runtime* runtime, const Domain& shape)
  : runtime_{runtime}, shape_{shape}
{
}

void RegionManager::destroy(bool unordered)
{
  for (auto& entry : entries_) entry.destroy(runtime_, unordered);
}

void RegionManager::push_entry()
{
  auto is = runtime_->find_or_create_index_space(shape_);
  auto fs = runtime_->create_field_space();

  entries_.emplace_back(runtime_->create_region(is, fs));
}

bool RegionManager::has_space() const { return !entries_.empty() && active_entry().has_space(); }

std::pair<Legion::LogicalRegion, Legion::FieldID> RegionManager::allocate_field(size_t field_size)
{
  if (!has_space()) push_entry();

  auto& entry = active_entry();
  auto fid =
    runtime_->allocate_field(entry.region.get_field_space(), entry.get_next_field_id(), field_size);
  return {entry.region, fid};
}

void RegionManager::import_region(const Legion::LogicalRegion& region)
{
  entries_.emplace_back(region);
}

}  // namespace legate::detail
