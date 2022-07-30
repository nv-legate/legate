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

#include <sstream>

#include "core/data/logical_store.h"
#include "core/partitioning/partition.h"
#include "core/runtime/launcher.h"
#include "core/runtime/req_analyzer.h"
#include "core/runtime/runtime.h"

namespace legate {

Partition::Partition(Runtime* runtime) : runtime_(runtime) {}

NoPartition::NoPartition(Runtime* runtime) : Partition(runtime) {}

bool NoPartition::is_complete_for(const LogicalStore* store) const { return false; }

bool NoPartition::is_disjoint_for(const LogicalStore* store) const { return false; }

Legion::LogicalPartition NoPartition::construct(Legion::LogicalRegion region,
                                                bool disjoint,
                                                bool complete) const
{
  return Legion::LogicalPartition::NO_PART;
}

std::unique_ptr<Projection> NoPartition::get_projection(LogicalStore store) const
{
  return std::make_unique<Projection>();
}

bool NoPartition::has_launch_domain() const { return false; }

Legion::Domain NoPartition::launch_domain() const
{
  assert(false);
  return Legion::Domain();
}

std::string NoPartition::to_string() const { return "NoPartition"; }

Tiling::Tiling(Runtime* runtime, Shape&& tile_shape, Shape&& color_shape, Shape&& offsets)
  : Partition(runtime),
    tile_shape_(std::forward<Shape>(tile_shape)),
    color_shape_(std::forward<Shape>(color_shape)),
    offsets_(std::forward<Shape>(offsets))
{
  if (offsets_.empty()) offsets_ = tuple<size_t>(tile_shape_.size(), 0);
  assert(tile_shape_.size() == color_shape_.size());
  assert(tile_shape_.size() == offsets_.size());
}

bool Tiling::operator==(const Tiling& other) const
{
  return tile_shape_ == other.tile_shape_ && color_shape_ == other.color_shape_ &&
         offsets_ == other.offsets_;
}

bool Tiling::operator<(const Tiling& other) const
{
  if (tile_shape_ < other.tile_shape_)
    return true;
  else if (other.tile_shape_ < tile_shape_)
    return false;
  if (color_shape_ < other.color_shape_)
    return true;
  else if (other.color_shape_ < color_shape_)
    return false;
  if (offsets_ < other.offsets_)
    return true;
  else
    return false;
}

bool Tiling::is_complete_for(const LogicalStore* store) const { return false; }

bool Tiling::is_disjoint_for(const LogicalStore* store) const { return true; }

Legion::LogicalPartition Tiling::construct(Legion::LogicalRegion region,
                                           bool disjoint,
                                           bool complete) const
{
  auto index_space     = region.get_index_space();
  auto index_partition = runtime_->partition_manager()->find_index_partition(index_space, *this);
  if (index_partition != Legion::IndexPartition::NO_PART)
    return runtime_->create_logical_partition(region, index_partition);

  auto ndim = static_cast<int32_t>(tile_shape_.size());

  Legion::DomainTransform transform;
  transform.m = ndim;
  transform.n = ndim;
  for (int32_t idx = 0; idx < ndim * ndim; ++idx) transform.matrix[idx] = 0;
  for (int32_t idx = 0; idx < ndim; ++idx) transform.matrix[ndim * idx + idx] = tile_shape_[idx];

  Legion::Domain extent;
  extent.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) {
    extent.rect_data[idx]        = offsets_[idx];
    extent.rect_data[idx + ndim] = tile_shape_[idx] - 1 + offsets_[idx];
  }

  Legion::Domain color_domain;
  color_domain.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) {
    color_domain.rect_data[idx]        = 0;
    color_domain.rect_data[idx + ndim] = color_shape_[idx] - 1;
  }

  auto color_space = runtime_->find_or_create_index_space(color_domain);

  auto kind = complete ? (disjoint ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_ALIASED_COMPLETE_KIND)
                       : (disjoint ? LEGION_DISJOINT_KIND : LEGION_ALIASED_KIND);

  index_partition =
    runtime_->create_restricted_partition(index_space, color_space, kind, transform, extent);
  runtime_->partition_manager()->record_index_partition(index_space, *this, index_partition);
  return runtime_->create_logical_partition(region, index_partition);
}

std::unique_ptr<Projection> Tiling::get_projection(LogicalStore store) const
{
  return store.find_or_create_partition(this);
}

bool Tiling::has_launch_domain() const { return true; }

Legion::Domain Tiling::launch_domain() const
{
  Legion::Domain launch_domain;
  int32_t ndim      = static_cast<int32_t>(color_shape_.size());
  launch_domain.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) {
    launch_domain.rect_data[idx]        = 0;
    launch_domain.rect_data[idx + ndim] = color_shape_[idx] - 1;
  }
  return launch_domain;
}

std::string Tiling::to_string() const
{
  std::stringstream ss;
  ss << "Tiling(tile:" << tile_shape_ << ",colors:" << color_shape_ << ",offset:" << offsets_
     << ")";
  return ss.str();
}

std::unique_ptr<Partition> create_tiling(Runtime* runtime,
                                         Shape&& tile_shape,
                                         Shape&& color_shape,
                                         Shape&& offsets /*= {}*/)
{
  return std::make_unique<Tiling>(runtime,
                                  std::forward<Shape>(tile_shape),
                                  std::forward<Shape>(color_shape),
                                  std::forward<Shape>(offsets));
}

std::unique_ptr<Partition> create_no_partition(Runtime* runtime)
{
  return std::make_unique<NoPartition>(runtime);
}

}  // namespace legate
