/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/runtime/detail/region_manager.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

void RegionManager::ManagerEntry::destroy(Runtime* runtime, bool unordered)
{
  runtime->destroy_region(region, unordered);
}

RegionManager::RegionManager(Runtime* runtime, const Domain& shape)
  : runtime_(runtime), shape_(shape)
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
  return std::make_pair(entry.region, fid);
}

void RegionManager::import_region(const Legion::LogicalRegion& region)
{
  entries_.emplace_back(region);
}

}  // namespace legate::detail
