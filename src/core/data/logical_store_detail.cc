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

#include "core/mapping/machine.h"
#include "core/runtime/partition_manager.h"
#include "core/runtime/req_analyzer.h"
#include "core/type/type_traits.h"
#include "core/utilities/buffer_builder.h"
#include "core/utilities/dispatch.h"
#include "legate_defines.h"

namespace legate {

extern Legion::Logger log_legate;

namespace detail {

////////////////////////////////////////////////////
// legate::detail::Storage
////////////////////////////////////////////////////

Storage::Storage(int32_t dim, std::unique_ptr<Type> type)
  : unbound_(true), dim_(dim), type_(std::move(type))
{
}

Storage::Storage(Shape extents, std::unique_ptr<Type> type, bool optimize_scalar)
  : dim_(extents.size()), extents_(extents), type_(std::move(type)), volume_(extents.volume())
{
  if (optimize_scalar && volume_ == 1) kind_ = Kind::FUTURE;
}

Storage::Storage(Shape extents, std::unique_ptr<Type> type, const Legion::Future& future)
  : dim_(extents.size()),
    extents_(extents),
    type_(std::move(type)),
    kind_(Kind::FUTURE),
    future_(future),
    volume_(extents.volume())
{
}

int32_t Storage::dim() { return dim_; }

LogicalRegionField* Storage::get_region_field()
{
#ifdef DEBUG_LEGATE
  assert(Kind::REGION_FIELD == kind_);
#endif
  if (nullptr == region_field_)
    region_field_ = Runtime::get_runtime()->create_region_field(extents_, type_->size());
  return region_field_.get();
}

Legion::Future Storage::get_future() const
{
#ifdef DEBUG_LEGATE
  assert(Kind::FUTURE == kind_);
#endif
  return future_;
}

void Storage::set_region_field(std::shared_ptr<LogicalRegionField>&& region_field)
{
  unbound_      = false;
  region_field_ = std::move(region_field);

  // TODO: this is a blocking operation
  auto domain = region_field_->domain();
  auto lo     = domain.lo();
  auto hi     = domain.hi();
  std::vector<size_t> extents;
  for (int32_t idx = 0; idx < lo.dim; ++idx) extents.push_back(hi[idx] - lo[idx] + 1);
  extents_ = extents;
}

void Storage::set_future(Legion::Future future) { future_ = future; }

RegionField Storage::map(LibraryContext* context)
{
#ifdef DEBUG_LEGATE
  assert(Kind::REGION_FIELD == kind_);
#endif
  return Runtime::get_runtime()->map_region_field(context, region_field_.get());
}

Partition* Storage::find_or_create_key_partition(const mapping::MachineDesc& machine,
                                                 const Restrictions& restrictions)
{
  uint32_t new_num_pieces = machine.count();
  if (num_pieces_ == new_num_pieces && key_partition_ != nullptr && restrictions_ == restrictions)
    return key_partition_.get();

  auto part_mgr     = Runtime::get_runtime()->partition_manager();
  auto launch_shape = part_mgr->compute_launch_shape(machine, restrictions, extents_);
  num_pieces_       = new_num_pieces;
  restrictions_     = restrictions;
  if (launch_shape.empty())
    key_partition_ = create_no_partition();
  else {
    auto tile_shape = part_mgr->compute_tile_shape(extents_, launch_shape);
    key_partition_  = create_tiling(std::move(tile_shape), std::move(launch_shape));
  }

  return key_partition_.get();
}

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

std::shared_ptr<StoragePartition> Storage::create_partition(std::shared_ptr<Partition> partition)
{
  return std::make_shared<StoragePartition>(shared_from_this(), std::move(partition));
}

////////////////////////////////////////////////////
// legate::detail::StoragePartition
////////////////////////////////////////////////////

StoragePartition::StoragePartition(std::shared_ptr<Storage> parent,
                                   std::shared_ptr<Partition> partition)
  : parent_(std::move(parent)), partition_(std::move(partition))
{
}

////////////////////////////////////////////////////
// legate::detail::LogicalStore
////////////////////////////////////////////////////

LogicalStore::LogicalStore(std::shared_ptr<Storage>&& storage)
  : store_id_(Runtime::get_runtime()->get_unique_store_id()),
    storage_(std::forward<decltype(storage_)>(storage)),
    transform_(std::make_shared<TransformStack>())
{
  if (!unbound()) extents_ = storage_->extents();
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);

  log_legate.debug() << "Create " << to_string();
#endif
}

LogicalStore::LogicalStore(Shape&& extents,
                           const std::shared_ptr<Storage>& storage,
                           std::shared_ptr<TransformStack>&& transform)
  : store_id_(Runtime::get_runtime()->get_unique_store_id()),
    extents_(std::forward<decltype(extents_)>(extents)),
    storage_(storage),
    transform_(std::forward<decltype(transform_)>(transform))
{
#ifdef DEBUG_LEGATE
  assert(transform_ != nullptr);

  log_legate.debug() << "Create " << to_string();
#endif
}

LogicalStore::~LogicalStore()
{
  if (mapped_ != nullptr) mapped_->unmap();
}

bool LogicalStore::unbound() const { return storage_->unbound(); }

const Shape& LogicalStore::extents() const
{
  if (extents_.empty()) {
    Runtime::get_runtime()->flush_scheduling_window();
    if (extents_.empty()) {
      throw std::invalid_argument("Illegal to access an uninitialized unbound store");
    }
  }
  return extents_;
}

size_t LogicalStore::volume() const { return extents().volume(); }

size_t LogicalStore::storage_size() const { return storage_->volume() * type().size(); }

int32_t LogicalStore::dim() const
{
  return unbound() ? storage_->dim() : static_cast<int32_t>(extents().size());
}

bool LogicalStore::has_scalar_storage() const { return storage_->kind() == Storage::Kind::FUTURE; }

const Type& LogicalStore::type() const { return storage_->type(); }

bool LogicalStore::transformed() const { return !transform_->identity(); }

LogicalRegionField* LogicalStore::get_region_field() { return storage_->get_region_field(); }

Legion::Future LogicalStore::get_future() { return storage_->get_future(); }

void LogicalStore::set_region_field(std::shared_ptr<LogicalRegionField>&& region_field)
{
#ifdef DEBUG_LEGATE
  assert(!has_scalar_storage());
#endif
  storage_->set_region_field(std::move(region_field));
  extents_ = storage_->extents();
}

void LogicalStore::set_future(Legion::Future future)
{
#ifdef DEBUG_LEGATE
  assert(has_scalar_storage());
#endif
  storage_->set_future(future);
}

std::shared_ptr<LogicalStore> LogicalStore::promote(int32_t extra_dim, size_t dim_size) const
{
  if (extra_dim < 0 || extra_dim > dim()) {
    throw std::invalid_argument("Invalid promotion on dimension " + std::to_string(extra_dim) +
                                " for a " + std::to_string(dim()) + "-D store");
  }

  auto new_extents = extents().insert(extra_dim, dim_size);
  auto transform   = transform_->push(std::make_unique<Promote>(extra_dim, dim_size));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<LogicalStore> LogicalStore::project(int32_t d, int64_t index) const
{
  if (d < 0 || d >= dim()) {
    throw std::invalid_argument("Invalid projection on dimension " + std::to_string(d) + " for a " +
                                std::to_string(dim()) + "-D store");
  } else if (index < 0 || index >= extents()[d]) {
    throw std::invalid_argument("Projection index " + std::to_string(index) +
                                " is out of bounds [0, " + std::to_string(extents()[d]) + ")");
  }

  auto new_extents = extents().remove(d);
  auto transform   = transform_->push(std::make_unique<Project>(d, index));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<LogicalStorePartition> LogicalStore::partition_by_tiling(Shape tile_shape)
{
  if (tile_shape.size() != extents().size()) {
    throw std::invalid_argument("Incompatible tile shape: expected a " +
                                std::to_string(extents().size()) + "-tuple, got a " +
                                std::to_string(tile_shape.size()) + "-tuple");
  }
  Shape color_shape(extents());
  // TODO: This better use std::transform
  for (size_t idx = 0; idx < tile_shape.size(); ++idx)
    color_shape[idx] = (color_shape[idx] + tile_shape[idx] - 1) / tile_shape[idx];
  auto partition = create_tiling(std::move(tile_shape), std::move(color_shape));
  return create_partition(std::move(partition));
}

std::shared_ptr<LogicalStore> LogicalStore::slice(int32_t idx, std::slice sl) const
{
  if (idx < 0 || idx >= dim()) {
    throw std::invalid_argument("Invalid slicing of dimension " + std::to_string(idx) + " for a " +
                                std::to_string(dim()) + "-D store");
  }

  auto start = sl.start();
  auto stop  = sl.size();
  auto step  = sl.stride();
  auto size  = extents()[idx];
  if (start < 0) { start = start + size; }
  if (stop < 0) { start = start + size; }

  if (step != 1) {
    throw std::invalid_argument("Unsupported slicing step: " + std::to_string(step));
  } else if (start >= size || stop > size) {
    throw std::invalid_argument("Out-of-bounds slicing on dimension " + std::to_string(idx) +
                                " for a store");
  }

  auto old_extents = extents();
  auto new_extents = Shape();
  for (int i = 0; i < idx; i++) { new_extents.append_inplace(old_extents[i]); }
  new_extents.append_inplace(stop - start);
  for (int i = idx + 1; i < old_extents.size(); i++) { new_extents.append_inplace(old_extents[i]); }

  auto transform =
    (start == 0) ? transform_ : transform_->push(std::make_unique<Shift>(idx, -start));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<LogicalStore> LogicalStore::transpose(std::vector<int32_t>&& axes) const
{
  if (axes.size() != dim()) {
    throw std::invalid_argument("Dimension Mismatch: expected " + std::to_string(dim()) +
                                " axes, but got " + std::to_string(axes.size()));
  } else if (axes.size() != (std::set<int32_t>(axes.begin(), axes.end())).size()) {
    throw std::invalid_argument("Duplicate axes found");
  }

  for (int i = 0; i < axes.size(); i++) {
    if (axes[i] < 0 || axes[i] >= dim()) {
      throw std::invalid_argument("Invalid axis " + std::to_string(axes[i]) + " for a " +
                                  std::to_string(dim()) + "-D store");
    }
  }

  auto old_extents = extents();
  auto new_extents = Shape();
  for (int i = 0; i < axes.size(); i++) { new_extents.append_inplace(old_extents[axes[i]]); }

  auto transform = transform_->push(std::make_unique<Transpose>(std::move(axes)));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<LogicalStore> LogicalStore::delinearize(int32_t idx,
                                                        std::vector<int64_t>&& sizes) const
{
  if (idx < 0 || idx >= dim()) {
    throw std::invalid_argument("Invalid delinearization on dimension " + std::to_string(idx) +
                                " for a " + std::to_string(dim()) + "-D store");
  }

  auto old_shape = extents();
  int64_t volume = 1;
  for (int i = 0; i < sizes.size(); i++) { volume *= sizes[i]; }

  if (old_shape[idx] != volume) {
    throw std::invalid_argument("Dimension of size " + std::to_string(old_shape[idx]) +
                                " cannot be delinearized into shape with volume " +
                                std::to_string(volume));
  }

  auto old_extents = extents();
  auto new_extents = Shape();
  for (int i = 0; i < idx; i++) { new_extents.append_inplace(old_extents[i]); }
  for (int i = 0; i < sizes.size(); i++) { new_extents.append_inplace(sizes[i]); }
  for (int i = idx + 1; i < old_extents.size(); i++) { new_extents.append_inplace(old_extents[i]); }

  auto transform = transform_->push(std::make_unique<Delinearize>(idx, std::move(sizes)));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  if (unbound()) { throw std::invalid_argument("Unbound store cannot be inlined mapped"); }
  if (nullptr != mapped_) return mapped_;
  if (storage_->kind() == Storage::Kind::FUTURE) {
    // TODO: future wrappers from inline mappings are read-only for now
    auto domain = to_domain(storage_->extents());
    FutureWrapper future(true, type().size(), domain, storage_->get_future());
    // Physical stores for future-backed stores shouldn't be cached, as they are not automatically
    // remapped to reflect changes by the runtime.
    return std::make_shared<Store>(dim(), type().clone(), -1, future, transform_);
  }

#ifdef DEBUG_LEGATE
  assert(storage_->kind() == Storage::Kind::REGION_FIELD);
#endif
  auto region_field = storage_->map(context);
  mapped_ = std::make_shared<Store>(dim(), type().clone(), -1, std::move(region_field), transform_);
  return mapped_;
}

Legion::ProjectionID LogicalStore::compute_projection(int32_t launch_ndim) const
{
  if (transform_->identity()) {
    if (launch_ndim != dim())
      return Runtime::get_runtime()->get_delinearizing_projection();
    else
      return 0;
  }

  auto ndim  = dim();
  auto point = transform_->invert(proj::create_symbolic_point(ndim));
  // TODO: We can't currently mix affine projections with delinearizing projections
#ifdef DEBUG_LEGATE
  assert(ndim == launch_ndim);
#endif
  return Runtime::get_runtime()->get_projection(ndim, point);
}

std::unique_ptr<Projection> LogicalStore::create_projection(const Partition* partition,
                                                            int32_t launch_ndim)
{
  if (has_scalar_storage()) return std::make_unique<Projection>();

  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  auto proj_id                        = compute_projection(launch_ndim);
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

std::shared_ptr<Partition> LogicalStore::find_or_create_key_partition(
  const mapping::MachineDesc& machine, const Restrictions& restrictions)
{
  uint32_t new_num_pieces = machine.count();
  if (num_pieces_ == new_num_pieces && key_partition_ != nullptr && restrictions_ == restrictions)
    return key_partition_;

  if (has_scalar_storage()) {
    num_pieces_    = new_num_pieces;
    restrictions_  = restrictions;
    key_partition_ = create_no_partition();
    return key_partition_;
  }

  Partition* storage_part =
    storage_->find_or_create_key_partition(machine, transform_->invert(restrictions));
  auto store_part = transform_->convert(storage_part);
  num_pieces_     = new_num_pieces;
  restrictions_   = restrictions;
  key_partition_  = std::move(store_part);
  return key_partition_;
}

void LogicalStore::set_key_partition(const Partition* partition)
{
  auto inverted = transform_->invert(partition);
  storage_->set_key_partition(std::move(inverted));
}

void LogicalStore::reset_key_partition() { storage_->reset_key_partition(); }

std::shared_ptr<LogicalStorePartition> LogicalStore::create_partition(
  std::shared_ptr<Partition> partition)
{
  if (unbound()) { throw std::invalid_argument("Unbound store cannot be manually partitioned"); }
  // TODO: the partition here should be inverted by the transform
  auto storage_partition = storage_->create_partition(partition);
  return std::make_shared<LogicalStorePartition>(std::move(storage_partition), shared_from_this());
}

void LogicalStore::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(has_scalar_storage());
  buffer.pack<bool>(unbound());
  buffer.pack<int32_t>(dim());
  type().pack(buffer);
  transform_->pack(buffer);
}

std::string LogicalStore::to_string() const
{
  std::stringstream ss;
  ss << "Store(" << store_id_ << ") {shape: ";
  if (unbound())
    ss << "(unbound)";
  else
    ss << extents();
  if (!transform_->identity())
    ss << ", transform: " << *transform_ << "}";
  else
    ss << "}";
  return std::move(ss).str();
}

////////////////////////////////////////////////////
// legate::detail::LogicalStorePartition
////////////////////////////////////////////////////
LogicalStorePartition::LogicalStorePartition(std::shared_ptr<StoragePartition> storage_partition,
                                             std::shared_ptr<LogicalStore> store)
  : storage_partition_(std::move(storage_partition)), store_(std::move(store))
{
}

}  // namespace detail
}  // namespace legate
