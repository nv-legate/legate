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
#include "legate_defines.h"

using namespace Legion;

namespace legate {

extern Logger log_legate;

LogicalRegionField::LogicalRegionField(Runtime* runtime, const LogicalRegion& lr, FieldID fid)
  : runtime_(runtime), lr_(lr), fid_(fid)
{
}

int32_t LogicalRegionField::dim() const { return lr_.get_dim(); }

Domain LogicalRegionField::domain() const
{
  return runtime_->get_index_space_domain(lr_.get_index_space());
}

namespace detail {

class LogicalStore {
 public:
  LogicalStore();
  LogicalStore(Runtime* runtime,
               LegateTypeCode code,
               std::vector<size_t> extents,
               std::shared_ptr<LogicalStore> parent,
               std::shared_ptr<StoreTransform> transform);

 public:
  LogicalStore(const LogicalStore& other) = default;
  LogicalStore& operator=(const LogicalStore& other) = default;

 public:
  LogicalStore(LogicalStore&& other) = default;
  LogicalStore& operator=(LogicalStore&& other) = default;

 public:
  int32_t dim() const;
  LegateTypeCode code() const;
  Legion::Domain domain() const;
  const std::vector<size_t>& extents() const;
  size_t volume() const;

 public:
  bool has_storage() const;
  std::shared_ptr<LogicalRegionField> get_storage();

 private:
  void create_storage();

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  std::unique_ptr<Projection> find_or_create_partition(const Partition* partition);
  std::unique_ptr<Partition> find_or_create_key_partition();

 private:
  Runtime* runtime_{nullptr};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  std::vector<size_t> extents_;
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  std::shared_ptr<LogicalStore> parent_{nullptr};
  std::shared_ptr<StoreTransform> transform_{nullptr};
  std::shared_ptr<Store> mapped_{nullptr};
};

LogicalStore::LogicalStore(Runtime* runtime,
                           LegateTypeCode code,
                           std::vector<size_t> extents,
                           std::shared_ptr<LogicalStore> parent,
                           std::shared_ptr<StoreTransform> transform)
  : runtime_(runtime), code_(code), extents_(std::move(extents)), transform_(std::move(transform))
{
}

int32_t LogicalStore::dim() const { return static_cast<int32_t>(extents_.size()); }

LegateTypeCode LogicalStore::code() const { return code_; }

Domain LogicalStore::domain() const
{
  assert(nullptr != region_field_);
  return region_field_->domain();
}

const std::vector<size_t>& LogicalStore::extents() const { return extents_; }

size_t LogicalStore::volume() const
{
  size_t vol = 1;
  for (auto extent : extents_) vol *= extent;
  return vol;
}

bool LogicalStore::has_storage() const { return nullptr != region_field_; }

std::shared_ptr<LogicalRegionField> LogicalStore::get_storage()
{
  if (!has_storage()) create_storage();
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
  auto lr = get_storage()->region();
  auto lp = partition->construct(
    lr, partition->is_disjoint_for(nullptr), partition->is_complete_for(nullptr));
  return std::make_unique<MapPartition>(lp, 0);
}

std::unique_ptr<Partition> LogicalStore::find_or_create_key_partition()
{
  auto part_mgr     = runtime_->get_partition_manager();
  auto launch_shape = part_mgr->compute_launch_shape(extents());
  if (launch_shape.empty())
    return create_no_partition(runtime_);
  else {
    auto tile_shape = part_mgr->compute_tile_shape(extents_, launch_shape);
    return create_tiling(runtime_, std::move(tile_shape), std::move(launch_shape));
  }
}

}  // namespace detail

LogicalStore::LogicalStore() {}

LogicalStore::LogicalStore(Runtime* runtime,
                           LegateTypeCode code,
                           std::vector<size_t> extents,
                           LogicalStore parent, /* = LogicalStore() */
                           std::shared_ptr<StoreTransform> transform /*= nullptr*/)
  : impl_(std::make_shared<detail::LogicalStore>(
      runtime, code, std::move(extents), parent.impl_, std::move(transform)))
{
}

int32_t LogicalStore::dim() const { return impl_->dim(); }

LegateTypeCode LogicalStore::code() const { return impl_->code(); }

Domain LogicalStore::domain() const { return impl_->domain(); }

const std::vector<size_t>& LogicalStore::extents() const { return impl_->extents(); }

size_t LogicalStore::volume() const { return impl_->volume(); }

std::shared_ptr<LogicalRegionField> LogicalStore::get_storage() { return impl_->get_storage(); }

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  return impl_->get_physical_store(context);
}

std::unique_ptr<Projection> LogicalStore::find_or_create_partition(const Partition* partition)
{
  return impl_->find_or_create_partition(partition);
}

std::unique_ptr<Partition> LogicalStore::find_or_create_key_partition()
{
  return impl_->find_or_create_key_partition();
}

}  // namespace legate
