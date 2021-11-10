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
#include "core/runtime/runtime.h"

namespace legate {

using Shape = std::vector<size_t>;

std::string to_string(const Shape& shape)
{
  std::stringstream ss;
  ss << "(";
  for (auto extent : shape) ss << extent << ",";
  ss << ")";
  return ss.str();
}

class NoPartition : public Partition {
 public:
  NoPartition(Runtime* runtime);

 public:
  virtual bool is_complete_for(const LogicalStore* store) const override;
  virtual bool is_disjoint_for(const LogicalStore* store) const override;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint,
                                             bool complete) const override;
  virtual std::unique_ptr<Projection> get_projection(LogicalStore store) const override;

 public:
  virtual bool has_launch_domain() const override;
  virtual Legion::Domain launch_domain() const override;

 public:
  virtual std::string to_string() const override;

 private:
  Runtime* runtime_;
};

class Tiling : public Partition {
 public:
  Tiling(Runtime* runtime, Shape&& tile_shape, Shape&& color_shape, Shape&& offsets);

 public:
  virtual bool is_complete_for(const LogicalStore* store) const override;
  virtual bool is_disjoint_for(const LogicalStore* store) const override;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint,
                                             bool complete) const override;
  virtual std::unique_ptr<Projection> get_projection(LogicalStore store) const override;

 public:
  virtual bool has_launch_domain() const override;
  virtual Legion::Domain launch_domain() const override;

 public:
  virtual std::string to_string() const override;

 private:
  Runtime* runtime_;
  Shape tile_shape_;
  Shape color_shape_;
  Shape offsets_;
};

struct PartitionByRestriction : public PartitioningFunctor {
 public:
  PartitionByRestriction(Legion::DomainTransform transform, Legion::Domain extent);

 public:
  virtual Legion::IndexPartition construct(Legion::Runtime* legion_runtime,
                                           Legion::Context legion_context,
                                           const Legion::IndexSpace& parent,
                                           const Legion::IndexSpace& color_space,
                                           Legion::PartitionKind kind) const override;

 private:
  Legion::DomainTransform transform_;
  Legion::Domain extent_;
};

Tiling::Tiling(Runtime* runtime, Shape&& tile_shape, Shape&& color_shape, Shape&& offsets)
  : runtime_(runtime),
    tile_shape_(std::forward<Shape>(tile_shape)),
    color_shape_(std::forward<Shape>(color_shape)),
    offsets_(std::forward<Shape>(offsets))
{
  if (offsets_.empty())
    for (auto _ : tile_shape_) offsets_.push_back(0);
  assert(tile_shape_.size() == color_shape_.size());
  assert(tile_shape_.size() == offsets_.size());
}

NoPartition::NoPartition(Runtime* runtime) : runtime_(runtime) {}

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
  return std::make_unique<Replicate>();
}

bool NoPartition::has_launch_domain() const { return false; }

Legion::Domain NoPartition::launch_domain() const
{
  assert(false);
  return Legion::Domain();
}

std::string NoPartition::to_string() const { return "NoPartition"; }

bool Tiling::is_complete_for(const LogicalStore* store) const { return false; }

bool Tiling::is_disjoint_for(const LogicalStore* store) const { return true; }

Legion::LogicalPartition Tiling::construct(Legion::LogicalRegion region,
                                           bool disjoint,
                                           bool complete) const
{
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
  auto index_space = region.get_index_space();

  auto kind = complete ? (disjoint ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_ALIASED_COMPLETE_KIND)
                       : (disjoint ? LEGION_DISJOINT_KIND : LEGION_ALIASED_KIND);

  PartitionByRestriction functor(transform, extent);
  auto index_partition = runtime_->create_index_partition(index_space, color_space, kind, &functor);
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
  ss << "Tiling(tile:" << legate::to_string(tile_shape_)
     << ",colors:" << legate::to_string(color_shape_) << ",offset:" << legate::to_string(offsets_)
     << ")";
  return ss.str();
}

PartitionByRestriction::PartitionByRestriction(Legion::DomainTransform transform,
                                               Legion::Domain extent)
  : transform_(transform), extent_(extent)
{
}

Legion::IndexPartition PartitionByRestriction::construct(Legion::Runtime* legion_runtime,
                                                         Legion::Context legion_context,
                                                         const Legion::IndexSpace& parent,
                                                         const Legion::IndexSpace& color_space,
                                                         Legion::PartitionKind kind) const
{
  return legion_runtime->create_partition_by_restriction(
    legion_context, parent, color_space, transform_, extent_, kind);
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
