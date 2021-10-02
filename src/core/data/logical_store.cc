/* Copyright 2021 NVIDIA Corporation
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

#include "core/data/logical_store.h"
#include "core/data/store.h"
#include "core/partitioning/partition.h"
#include "core/runtime/launcher.h"
#include "core/runtime/runtime.h"

using namespace Legion;

namespace legate {

LogicalRegionField::LogicalRegionField(Runtime* runtime, const LogicalRegion& lr, FieldID fid)
  : runtime_(runtime), lr_(lr), fid_(fid)
{
}

int32_t LogicalRegionField::dim() const { return lr_.get_dim(); }

Domain LogicalRegionField::domain() const
{
  return runtime_->get_index_space_domain(lr_.get_index_space());
}

LogicalStore::LogicalStore(Runtime* runtime,
                           LegateTypeCode code,
                           std::vector<size_t> extents,
                           std::shared_ptr<StoreTransform> transform /*= nullptr*/)
  : runtime_(runtime), code_(code), extents_(std::move(extents)), transform_(std::move(transform))
{
}

int32_t LogicalStore::dim() const { return static_cast<int32_t>(extents_.size()); }

Domain LogicalStore::domain() const
{
  assert(nullptr != region_field_);
  return region_field_->domain();
}

size_t LogicalStore::volume() const
{
  size_t vol = 1;
  for (auto extent : extents_) vol *= extent;
  return vol;
}

std::shared_ptr<LogicalRegionField> LogicalStore::get_storage()
{
  if (!has_storage()) create_storage();
  return region_field_;
}

std::shared_ptr<LogicalRegionField> LogicalStore::get_storage_unsafe() const
{
  return region_field_;
}

void LogicalStore::create_storage()
{
  region_field_ = runtime_->create_region_field(extents_, code_);
}

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  if (nullptr != mapped_) return mapped_;
  auto rf = runtime_->map_region_field(context, region_field_);
  mapped_ = std::make_shared<Store>(dim(), code_, -1, std::move(rf), transform_);
  return mapped_;
}

std::unique_ptr<Projection> LogicalStore::find_or_create_partition(const Partition* partition)
{
  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  if (!has_storage()) create_storage();
  auto lp =
    partition->construct(this, partition->is_disjoint_for(this), partition->is_complete_for(this));
  return std::make_unique<MapPartition>(lp, 0);
}

std::unique_ptr<Partition> LogicalStore::find_or_create_key_partition()
{
  auto part_mgr     = runtime_->get_partition_manager();
  auto launch_shape = part_mgr->compute_launch_shape(this);
  if (launch_shape.empty())
    return create_no_partition(runtime_);
  else {
    auto tile_shape = part_mgr->compute_tile_shape(extents_, launch_shape);
    return create_tiling(runtime_, std::move(tile_shape), std::move(launch_shape));
  }
}

}  // namespace legate
