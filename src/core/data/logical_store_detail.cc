/* Copyright 2022 NVIDIA Corporation
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

#include "core/data/logical_store_detail.h"

#include "core/runtime/req_analyzer.h"
#include "core/utilities/buffer_builder.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/type_traits.h"
#include "legate_defines.h"

namespace legate {

extern Legion::Logger log_legate;

namespace detail {

////////////////////////////////////////////////////
// legate::detail::Storage
////////////////////////////////////////////////////

Storage::Storage(tuple<size_t> extents, LegateTypeCode code)
  : extents_(extents), code_(code), volume_(extents.volume())
{
}

Storage::Storage(tuple<size_t> extents, LegateTypeCode code, const Legion::Future& future)
  : extents_(extents), code_(code), kind_(Kind::FUTURE), future_(future), volume_(extents.volume())
{
}

int32_t Storage::dim() { return static_cast<int32_t>(extents_.size()); }

LogicalRegionField* Storage::get_region_field()
{
#ifdef DEBUG_LEGATE
  assert(Kind::REGION_FIELD == kind_);
#endif
  if (nullptr == region_field_)
    region_field_ = Runtime::get_runtime()->create_region_field(extents_, code_);
  return region_field_.get();
}

Legion::Future Storage::get_future() const
{
#ifdef DEBUG_LEGATE
  assert(Kind::FUTURE == kind_);
#endif
  return future_;
}

RegionField Storage::map(LibraryContext* context)
{
#ifdef DEBUG_LEGATE
  assert(Kind::REGION_FIELD == kind_);
#endif
  return Runtime::get_runtime()->map_region_field(context, region_field_.get());
}

Partition* Storage::find_key_partition() const { return key_partition_.get(); }

void Storage::set_key_partition(std::unique_ptr<Partition>&& key_partition)
{
  key_partition_ = std::forward<decltype(key_partition_)>(key_partition);
}

void Storage::reset_key_partition() { key_partition_ = nullptr; }

Legion::LogicalPartition Storage::find_or_create_legion_partition(const Partition* partition)
{
#ifdef DEBUG_LEGATE
  assert(Kind::REGION_FIELD == kind_);
#endif
  auto region = get_region_field()->region();
  return partition->construct(
    region, partition->is_disjoint_for(nullptr), partition->is_complete_for(nullptr));
}

////////////////////////////////////////////////////
// legate::detail::LogicalStore
////////////////////////////////////////////////////

LogicalStore::LogicalStore(std::shared_ptr<Storage>&& storage)
  : store_id_(Runtime::get_runtime()->get_unique_store_id()),
    extents_(storage->extents()),
    storage_(std::forward<decltype(storage_)>(storage)),
    transform_(std::make_shared<TransformStack>())
{
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);

  log_legate.debug() << "Create Store(" << store_id_ << ") {shape: " << extents_ << "}";
#endif
}

LogicalStore::LogicalStore(tuple<size_t>&& extents,
                           const std::shared_ptr<Storage>& storage,
                           std::shared_ptr<TransformStack>&& transform)
  : store_id_(Runtime::get_runtime()->get_unique_store_id()),
    extents_(std::forward<decltype(extents_)>(extents)),
    storage_(storage),
    transform_(std::forward<decltype(transform_)>(transform))
{
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);

  log_legate.debug() << "Create Store(" << store_id_ << ") {shape: " << extents_
                     << ", transform: " << *transform_ << "}";
#endif
}

LogicalStore::~LogicalStore()
{
  if (mapped_ != nullptr) mapped_->unmap();
}

const tuple<size_t>& LogicalStore::extents() const { return extents_; }

size_t LogicalStore::volume() const { return extents_.volume(); }

int32_t LogicalStore::dim() const { return static_cast<int32_t>(extents_.size()); }

bool LogicalStore::scalar() const { return storage_->kind() == Storage::Kind::FUTURE; }

LegateTypeCode LogicalStore::code() const { return storage_->code(); }

LogicalRegionField* LogicalStore::get_region_field() { return storage_->get_region_field(); }

Legion::Future LogicalStore::get_future() { return storage_->get_future(); }

std::shared_ptr<LogicalStore> LogicalStore::promote(int32_t extra_dim,
                                                    size_t dim_size,
                                                    std::shared_ptr<LogicalStore> parent) const
{
  if (extra_dim < 0 || extra_dim > dim()) {
    log_legate.error("Invalid promotion on dimension %d for a %d-D store", extra_dim, dim());
    LEGATE_ABORT;
  }

  auto new_extents = extents_.insert(extra_dim, dim_size);
  auto transform   = transform_->push(std::make_unique<Promote>(extra_dim, dim_size));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  // TODO: Need to support inline mapping for scalars
  assert(storage_->kind() == Storage::Kind::REGION_FIELD);
  if (nullptr != mapped_) return mapped_;
  auto region_field = storage_->map(context);
  mapped_ = std::make_shared<Store>(dim(), code(), -1, std::move(region_field), transform_);
  return std::move(mapped_);
}

Legion::ProjectionID LogicalStore::compute_projection() const
{
  if (transform_->identity()) return 0;

  auto ndim  = dim();
  auto point = transform_->invert(proj::create_symbolic_point(ndim));
  return Runtime::get_runtime()->get_projection(ndim, point);
}

std::unique_ptr<Projection> LogicalStore::create_projection(const Partition* partition)
{
  if (scalar()) return std::make_unique<Projection>();

  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  auto proj_id                        = compute_projection();
  auto* orig_partition                = partition;
  std::unique_ptr<Partition> inverted = nullptr;
  if (!transform_->identity()) {
    inverted  = transform_->invert(partition);
    partition = inverted.get();
  }

#ifdef DEBUG_LEGATE
  log_legate.debug() << "Partition Store(" << store_id_ << ") {partition: " << *orig_partition
                     << ", inverted: " << *partition << ", projection: " << proj_id << "}";
#endif

  auto region_field     = get_region_field();
  auto region           = region_field->region();
  auto legion_partition = partition->construct(
    region, partition->is_disjoint_for(nullptr), partition->is_complete_for(nullptr));
  return std::make_unique<Projection>(legion_partition, proj_id);
}

std::unique_ptr<Partition> LogicalStore::find_or_create_key_partition()
{
  if (scalar()) return create_no_partition();

  auto part_mgr     = Runtime::get_runtime()->partition_manager();
  auto launch_shape = part_mgr->compute_launch_shape(extents_);
  if (launch_shape.empty())
    return create_no_partition();
  else {
    auto tile_shape = part_mgr->compute_tile_shape(extents_, launch_shape);
    return create_tiling(std::move(tile_shape), std::move(launch_shape));
  }
}

void LogicalStore::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(scalar());
  buffer.pack<bool>(false);
  buffer.pack<int32_t>(dim());
  buffer.pack<int32_t>(code());
  transform_->pack(buffer);
}

}  // namespace detail
}  // namespace legate
