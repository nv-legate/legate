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

#include "core/data/detail/logical_store.h"

#include "core/data/detail/store.h"
#include "core/data/detail/transform.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/operation.h"
#include "core/operation/detail/projection.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"
#include "core/type/type_traits.h"
#include "core/utilities/dispatch.h"

#include "legate_defines.h"

#include <utility>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::detail::Storage
////////////////////////////////////////////////////

Storage::Storage(int32_t dim, std::shared_ptr<Type> type)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    unbound_{true},
    dim_{dim},
    type_{std::move(type)},
    offsets_{legate::full(dim_, size_t{0})}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { log_legate.debug() << "Create " << to_string(); }
}

Storage::Storage(const Shape& extents, std::shared_ptr<Type> type, bool optimize_scalar)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    dim_{static_cast<std::int32_t>(extents.size())},
    extents_{extents},
    type_{std::move(type)},
    offsets_{legate::full(dim_, size_t{0})}
{
  if (optimize_scalar && extents_.volume() == 1) kind_ = Kind::FUTURE;
  if (LegateDefined(LEGATE_USE_DEBUG)) { log_legate.debug() << "Create " << to_string(); }
}

Storage::Storage(const Shape& extents, std::shared_ptr<Type> type, const Legion::Future& future)
  : storage_id_(Runtime::get_runtime()->get_unique_storage_id()),
    dim_(extents.size()),
    extents_(extents),
    type_(std::move(type)),
    kind_(Kind::FUTURE),
    future_{std::make_unique<Legion::Future>(future)},
    offsets_{legate::full(dim_, size_t{0})}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { log_legate.debug() << "Create " << to_string(); }
}

Storage::Storage(Shape&& extents,
                 std::shared_ptr<Type> type,
                 std::shared_ptr<StoragePartition> parent,
                 Shape&& color,
                 Shape&& offsets)
  : storage_id_(Runtime::get_runtime()->get_unique_storage_id()),
    dim_(extents.size()),
    extents_(std::move(extents)),
    type_(std::move(type)),
    level_(parent->level() + 1),
    parent_(std::move(parent)),
    color_(std::move(color)),
    offsets_(std::move(offsets))
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { log_legate.debug() << "Create " << to_string(); }
}

Storage::~Storage()
{
  if (!Runtime::get_runtime()->initialized()) {
    // FIXME: Leak the Future handle if the runtime has already shut down, as there's no hope that
    // this would be collected by the Legion runtime
    future_.release();
  }
}

const Shape& Storage::extents() const
{
  if (unbound_) {
    Runtime::get_runtime()->flush_scheduling_window();
    if (unbound_) throw std::invalid_argument("Illegal to access an uninitialized unbound store");
  }
  return extents_;
}

const Shape& Storage::offsets() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(!unbound_); }
  return offsets_;
}

bool Storage::overlaps(const std::shared_ptr<Storage>& other) const
{
  const auto* lhs = this;
  const auto* rhs = other.get();

  if (lhs == rhs) return true;

  if (lhs->get_root() != rhs->get_root()) return false;

  if (lhs->volume() == 0 || rhs->volume() == 0) return false;

  for (int32_t idx = 0; idx < dim_; ++idx) {
    auto loff = lhs->offsets_[idx];
    auto lext = lhs->extents_[idx];
    auto roff = rhs->offsets_[idx];
    auto rext = rhs->extents_[idx];

    if (loff <= roff ? roff < loff + lext : loff < roff + rext) { continue; }
    return false;
  }
  return true;
}

size_t Storage::volume() const { return extents().volume(); }

std::shared_ptr<Storage> Storage::slice(Shape tile_shape, Shape offsets)
{
  if (Kind::FUTURE == kind_) return shared_from_this();

  auto root  = get_root();
  auto shape = root->extents();

  auto can_tile_completely =
    (shape % tile_shape).sum() == 0 && (offsets % tile_shape).sum() == 0 &&
    Runtime::get_runtime()->partition_manager()->use_complete_tiling(shape, tile_shape);

  Shape color_shape, color;
  tuple<int64_t> signed_offsets;
  if (can_tile_completely) {
    color_shape    = shape / tile_shape;
    color          = offsets / tile_shape;
    signed_offsets = legate::full<int64_t>(shape.size(), 0);  // tuple<int64_t>(0, shape.size());
  } else {
    color_shape    = legate::full<size_t>(shape.size(), 1);
    color          = legate::full<size_t>(shape.size(), 0);
    signed_offsets = apply([](size_t v) { return static_cast<int64_t>(v); }, offsets);
  }

  auto tiling =
    create_tiling(std::move(tile_shape), std::move(color_shape), std::move(signed_offsets));
  auto storage_partition = root->create_partition(std::move(tiling), can_tile_completely);
  return storage_partition->get_child_storage(color);
}

std::shared_ptr<const Storage> Storage::get_root() const
{
  return nullptr == parent_ ? shared_from_this() : parent_->get_root();
}

std::shared_ptr<Storage> Storage::get_root()
{
  return nullptr == parent_ ? shared_from_this() : parent_->get_root();
}

std::shared_ptr<LogicalRegionField> Storage::get_region_field()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(Kind::REGION_FIELD == kind_); }
  if (region_field_ != nullptr) return region_field_;

  if (nullptr == parent_) {
    region_field_ = Runtime::get_runtime()->create_region_field(extents_, type_->size());
    if (destroyed_out_of_order_) region_field_->allow_out_of_order_destruction();
  } else
    region_field_ = parent_->get_child_data(color_);

  return region_field_;
}

Legion::Future Storage::get_future() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(Kind::FUTURE == kind_); }
  return future_ != nullptr ? *future_ : Legion::Future{};
}

void Storage::set_region_field(std::shared_ptr<LogicalRegionField>&& region_field)
{
  assert(unbound_ && region_field_ == nullptr);
  assert(parent_ == nullptr);

  unbound_      = false;
  region_field_ = std::move(region_field);
  if (destroyed_out_of_order_) region_field_->allow_out_of_order_destruction();

  // TODO: this is a blocking operation
  auto domain = region_field_->domain();
  auto lo     = domain.lo();
  auto hi     = domain.hi();
  std::vector<size_t> extents;
  for (int32_t idx = 0; idx < lo.dim; ++idx) extents.push_back(hi[idx] - lo[idx] + 1);
  extents_ = Shape{std::move(extents)};
}

void Storage::set_future(Legion::Future future)
{
  future_ = std::make_unique<Legion::Future>(std::move(future));
}

RegionField Storage::map()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(Kind::REGION_FIELD == kind_); }
  auto region_field = get_region_field();
  auto mapped       = region_field->map();
  // Set the right subregion so the physical store can see the right domain
  mapped.set_logical_region(region_field->region());
  return mapped;
}

void Storage::allow_out_of_order_destruction()
{
  // Technically speaking this property only needs to be tracked on (root) LogicalRegionFields, but
  // a Storage may not have instantiated its region_field_ yet, so we note this also on the (root)
  // Storage, in case we need to propagate later. We only need to note this on the root Storage,
  // because any call that sets region_field_ (get_region_field(), set_region_field()) will end up
  // touching the root Storage.
  if (parent_ != nullptr)
    get_root()->allow_out_of_order_destruction();
  else if (!destroyed_out_of_order_) {
    destroyed_out_of_order_ = true;
    if (region_field_ != nullptr) region_field_->allow_out_of_order_destruction();
  }
}

Restrictions Storage::compute_restrictions() const
{
  return legate::full<Restriction>(dim_, Restriction::ALLOW);
}

Partition* Storage::find_key_partition(const mapping::detail::Machine& machine,
                                       const Restrictions& restrictions) const
{
  uint32_t new_num_pieces = machine.count();
  if (num_pieces_ == new_num_pieces && key_partition_ != nullptr &&
      key_partition_->satisfies_restrictions(restrictions))
    return key_partition_.get();
  else if (parent_ != nullptr)
    return parent_->find_key_partition(machine, restrictions);
  else
    return nullptr;
}

void Storage::set_key_partition(const mapping::detail::Machine& machine,
                                std::unique_ptr<Partition>&& key_partition)
{
  num_pieces_    = machine.count();
  key_partition_ = std::move(key_partition);
}

void Storage::reset_key_partition() { key_partition_ = nullptr; }

std::shared_ptr<StoragePartition> Storage::create_partition(std::shared_ptr<Partition> partition,
                                                            std::optional<bool> complete)
{
  if (!complete.has_value()) complete = partition->is_complete_for(this);
  return std::make_shared<StoragePartition>(
    shared_from_this(), std::move(partition), complete.value());
}

std::string Storage::to_string() const
{
  std::stringstream ss;

  ss << "Storage(" << storage_id_ << ") {shape: ";
  if (unbound_)
    ss << "(unbound)";
  else
    ss << extents_;
  ss << ", dim: " << dim_ << ", kind: " << (kind_ == Kind::REGION_FIELD ? "Region" : "Future")
     << ", type: " << type_->to_string() << ", level: " << level_ << "}";

  return std::move(ss).str();
}

////////////////////////////////////////////////////
// legate::detail::StoragePartition
////////////////////////////////////////////////////

StoragePartition::StoragePartition(std::shared_ptr<Storage> parent,
                                   std::shared_ptr<Partition> partition,
                                   bool complete)
  : complete_(complete),
    level_(parent->level() + 1),
    parent_(std::move(parent)),
    partition_(std::move(partition))
{
}

std::shared_ptr<const Storage> StoragePartition::get_root() const { return parent_->get_root(); }

std::shared_ptr<Storage> StoragePartition::get_root() { return parent_->get_root(); }

std::shared_ptr<Storage> StoragePartition::get_child_storage(const Shape& color)
{
  if (partition_->kind() != Partition::Kind::TILING)
    throw std::runtime_error("Sub-storage is implemented only for tiling");
  auto tiling        = static_cast<Tiling*>(partition_.get());
  auto child_extents = tiling->get_child_extents(parent_->extents(), color);
  auto child_offsets = tiling->get_child_offsets(color);
  return std::make_shared<Storage>(std::move(child_extents),
                                   parent_->type(),
                                   shared_from_this(),
                                   Shape(color),
                                   std::move(child_offsets));
}

std::shared_ptr<LogicalRegionField> StoragePartition::get_child_data(const Shape& color)
{
  if (partition_->kind() != Partition::Kind::TILING)
    throw std::runtime_error("Sub-storage is implemented only for tiling");
  auto tiling = static_cast<Tiling*>(partition_.get());
  return parent_->get_region_field()->get_child(tiling, color, complete_);
}

Partition* StoragePartition::find_key_partition(const mapping::detail::Machine& machine,
                                                const Restrictions& restrictions) const
{
  return parent_->find_key_partition(machine, restrictions);
}

Legion::LogicalPartition StoragePartition::get_legion_partition()
{
  return parent_->get_region_field()->get_legion_partition(partition_.get(), complete_);
}

bool StoragePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return partition_->is_disjoint_for(launch_domain);
}

////////////////////////////////////////////////////
// legate::detail::LogicalStore
////////////////////////////////////////////////////

void assert_fixed_storage_size(std::shared_ptr<Storage> storage)
{
  if (storage->type()->variable_size()) {
    throw std::invalid_argument("Store cannot be created with variable size type " +
                                storage->type()->to_string());
  }
}

LogicalStore::LogicalStore(std::shared_ptr<Storage>&& storage)
  : store_id_(Runtime::get_runtime()->get_unique_store_id()),
    storage_(std::move(storage)),
    transform_(std::make_shared<TransformStack>())
{
  assert_fixed_storage_size(storage_);
  if (!unbound()) extents_ = storage_->extents();
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(transform_ != nullptr);

    log_legate.debug() << "Create " << to_string();
  }
}

LogicalStore::LogicalStore(Shape&& extents,
                           const std::shared_ptr<Storage>& storage,
                           std::shared_ptr<TransformStack>&& transform)
  : store_id_(Runtime::get_runtime()->get_unique_store_id()),
    extents_(std::move(extents)),
    storage_(storage),
    transform_(std::move(transform))
{
  assert_fixed_storage_size(storage_);
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(transform_ != nullptr);

    log_legate.debug() << "Create " << to_string();
  }
}

bool LogicalStore::unbound() const { return storage_->unbound(); }

const Shape& LogicalStore::extents() const
{
  if (unbound()) {
    Runtime::get_runtime()->flush_scheduling_window();
    if (unbound()) throw std::invalid_argument("Illegal to access an uninitialized unbound store");
  }
  return extents_;
}

size_t LogicalStore::volume() const { return extents().volume(); }

size_t LogicalStore::storage_size() const { return storage_->volume() * type()->size(); }

int32_t LogicalStore::dim() const
{
  return unbound() ? storage_->dim() : static_cast<int32_t>(extents().size());
}

bool LogicalStore::overlaps(const std::shared_ptr<LogicalStore>& other) const
{
  return storage_->overlaps(other->storage_);
}

bool LogicalStore::has_scalar_storage() const { return storage_->kind() == Storage::Kind::FUTURE; }

std::shared_ptr<Type> LogicalStore::type() const { return storage_->type(); }

bool LogicalStore::transformed() const { return !transform_->identity(); }

uint64_t LogicalStore::id() const { return store_id_; }

const Storage* LogicalStore::get_storage() const { return storage_.get(); }

std::shared_ptr<LogicalRegionField> LogicalStore::get_region_field()
{
  return storage_->get_region_field();
}

Legion::Future LogicalStore::get_future() { return storage_->get_future(); }

void LogicalStore::set_region_field(std::shared_ptr<LogicalRegionField>&& region_field)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(!has_scalar_storage()); }
  storage_->set_region_field(std::move(region_field));
  extents_ = storage_->extents();
}

void LogicalStore::set_future(Legion::Future future)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(has_scalar_storage()); }
  storage_->set_future(future);
}

std::shared_ptr<LogicalStore> LogicalStore::promote(int32_t extra_dim, size_t dim_size)
{
  if (extra_dim < 0 || extra_dim > dim()) {
    throw std::invalid_argument("Invalid promotion on dimension " + std::to_string(extra_dim) +
                                " for a " + std::to_string(dim()) + "-D store");
  }

  auto new_extents = extents().insert(extra_dim, dim_size);
  auto transform   = transform_->push(std::make_unique<Promote>(extra_dim, dim_size));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<LogicalStore> LogicalStore::project(int32_t d, int64_t index)
{
  auto old_extents = extents();

  if (d < 0 || d >= dim()) {
    throw std::invalid_argument("Invalid projection on dimension " + std::to_string(d) + " for a " +
                                std::to_string(dim()) + "-D store");
  } else if (index < 0 || index >= static_cast<int64_t>(old_extents[d])) {
    throw std::invalid_argument("Projection index " + std::to_string(index) +
                                " is out of bounds [0, " + std::to_string(old_extents[d]) + ")");
  }

  auto new_extents = old_extents.remove(d);
  auto transform   = transform_->push(std::make_unique<Project>(d, index));
  auto substorage =
    volume() == 0
      ? storage_
      : storage_->slice(transform->invert_extents(new_extents),
                        transform->invert_point(legate::full<size_t>(new_extents.size(), 0)));
  return std::make_shared<LogicalStore>(
    std::move(new_extents), std::move(substorage), std::move(transform));
}

std::shared_ptr<LogicalStorePartition> LogicalStore::partition_by_tiling(Shape tile_shape)
{
  if (tile_shape.size() != extents().size()) {
    throw std::invalid_argument("Incompatible tile shape: expected a " +
                                std::to_string(extents().size()) + "-tuple, got a " +
                                std::to_string(tile_shape.size()) + "-tuple");
  }
  if (tile_shape.volume() == 0) {
    throw std::invalid_argument("Tile shape must have a volume greater than 0");
  }
  auto color_shape = apply([](auto c, auto t) { return (c + t - 1) / t; }, extents(), tile_shape);
  auto partition   = create_tiling(std::move(tile_shape), std::move(color_shape));
  return create_partition(std::move(partition), true);
}

std::shared_ptr<LogicalStore> LogicalStore::slice(int32_t idx, Slice slice)
{
  if (idx < 0 || idx >= dim()) {
    throw std::invalid_argument("Invalid slicing of dimension " + std::to_string(idx) + " for a " +
                                std::to_string(dim()) + "-D store");
  }

  auto sanitize_slice = [](const Slice& slice, size_t extent) {
    int64_t start = slice.start.value_or(0);
    int64_t stop  = slice.stop.value_or(extent);

    if (start < 0) start += extent;
    if (stop < 0) stop += extent;

    return std::make_pair<size_t, size_t>(std::max<int64_t>(0, start), std::max<int64_t>(0, stop));
  };

  auto exts          = extents();
  auto [start, stop] = sanitize_slice(slice, exts[idx]);

  if (start < stop && (start >= exts[idx] || stop > exts[idx])) {
    throw std::invalid_argument("Out-of-bounds slicing on dimension " + std::to_string(dim()) +
                                " for a store of shape " + extents().to_string());
  }

  exts[idx] = stop - start;

  if (exts[idx] == extents()[idx]) return shared_from_this();

  auto transform =
    (start == 0) ? transform_ : transform_->push(std::make_unique<Shift>(idx, -start));
  auto substorage =
    volume() == 0 ? storage_
                  : storage_->slice(transform->invert_extents(exts),
                                    transform->invert_point(legate::full<size_t>(exts.size(), 0)));
  return std::make_shared<LogicalStore>(
    std::move(exts), std::move(substorage), std::move(transform));
}

std::shared_ptr<LogicalStore> LogicalStore::transpose(const std::vector<int32_t>& axes)
{
  return transpose(std::vector<int32_t>(axes));
}

std::shared_ptr<LogicalStore> LogicalStore::transpose(std::vector<int32_t>&& axes)
{
  if (axes.size() != static_cast<size_t>(dim())) {
    throw std::invalid_argument("Dimension Mismatch: expected " + std::to_string(dim()) +
                                " axes, but got " + std::to_string(axes.size()));
  } else if (axes.size() != (std::set<int32_t>(axes.begin(), axes.end())).size()) {
    throw std::invalid_argument("Duplicate axes found");
  }

  for (auto&& ax : axes) {
    if (ax < 0 || ax >= dim()) {
      throw std::invalid_argument("Invalid axis " + std::to_string(ax) + " for a " +
                                  std::to_string(dim()) + "-D store");
    }
  }

  auto old_extents = extents();
  auto new_extents = Shape();
  for (auto&& ax : axes) { new_extents.append_inplace(old_extents[ax]); }

  auto transform = transform_->push(std::make_unique<Transpose>(std::move(axes)));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<LogicalStore> LogicalStore::delinearize(int32_t idx,
                                                        const std::vector<int64_t>& sizes)
{
  return delinearize(idx, std::vector<int64_t>(sizes));
}

std::shared_ptr<LogicalStore> LogicalStore::delinearize(int32_t idx, std::vector<int64_t>&& sizes)
{
  if (idx < 0 || idx >= dim()) {
    throw std::invalid_argument("Invalid delinearization on dimension " + std::to_string(idx) +
                                " for a " + std::to_string(dim()) + "-D store");
  }

  auto old_shape = extents();
  int64_t volume = 1;
  for (auto&& size : sizes) volume *= size;

  if (static_cast<int64_t>(old_shape[idx]) != volume) {
    throw std::invalid_argument("Dimension of size " + std::to_string(old_shape[idx]) +
                                " cannot be delinearized into shape with volume " +
                                std::to_string(volume));
  }

  auto old_extents = extents();
  auto new_extents = Shape();
  for (int i = 0; i < idx; i++) { new_extents.append_inplace(old_extents[i]); }
  for (auto&& size : sizes) { new_extents.append_inplace(size); }
  for (int i = idx + 1; static_cast<size_t>(i) < old_extents.size(); i++) {
    new_extents.append_inplace(old_extents[i]);
  }

  auto transform = transform_->push(std::make_unique<Delinearize>(idx, std::move(sizes)));
  return std::make_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

std::shared_ptr<Store> LogicalStore::get_physical_store()
{
  if (unbound()) { throw std::invalid_argument("Unbound store cannot be inlined mapped"); }
  if (nullptr != mapped_) return mapped_;
  if (storage_->kind() == Storage::Kind::FUTURE) {
    // TODO: future wrappers from inline mappings are read-only for now
    auto domain = to_domain(storage_->extents());
    FutureWrapper future(true, type()->size(), domain, storage_->get_future());
    // Physical stores for future-backed stores shouldn't be cached, as they are not automatically
    // remapped to reflect changes by the runtime.
    return std::make_shared<Store>(dim(), type(), -1, std::move(future), transform_);
  }

  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(storage_->kind() == Storage::Kind::REGION_FIELD); }
  auto region_field = storage_->map();
  mapped_ = std::make_shared<Store>(dim(), type(), -1, std::move(region_field), transform_);
  return mapped_;
}

void LogicalStore::detach()
{
  if (transformed()) throw std::invalid_argument("Manual detach must be called on the root store");
  if (has_scalar_storage() || unbound())
    throw std::invalid_argument("Only stores created with share=true can be manually detached");
  get_region_field()->detach();
}

void LogicalStore::allow_out_of_order_destruction()
{
  if (Runtime::get_runtime()->consensus_match_required())
    storage_->allow_out_of_order_destruction();
}

Restrictions LogicalStore::compute_restrictions() const
{
  return transform_->convert(storage_->compute_restrictions());
}

Legion::ProjectionID LogicalStore::compute_projection(
  int32_t launch_ndim, std::optional<proj::SymbolicFunctor> proj_fn) const
{
  auto ndim = dim();

  if (proj_fn != nullptr) {
    assert(!transformed());
    auto point = proj_fn.value()(proj::create_symbolic_point(launch_ndim));
    return Runtime::get_runtime()->get_projection(launch_ndim, point);
  }

  if (transform_->identity()) {
    if (launch_ndim != dim())
      return Runtime::get_runtime()->get_delinearizing_projection();
    else
      return 0;
  }

  auto point = transform_->invert(proj::create_symbolic_point(ndim));
  // TODO: We can't currently mix affine projections with delinearizing projections
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(ndim == launch_ndim); }
  return Runtime::get_runtime()->get_projection(ndim, point);
}

std::shared_ptr<Partition> LogicalStore::find_or_create_key_partition(
  const mapping::detail::Machine& machine, const Restrictions& restrictions)
{
  uint32_t new_num_pieces = machine.count();
  if (num_pieces_ == new_num_pieces && key_partition_ != nullptr &&
      key_partition_->satisfies_restrictions(restrictions))
    return key_partition_;

  if (has_scalar_storage() || extents_.empty() || volume() == 0) { return create_no_partition(); }

  Partition* storage_part = nullptr;
  if (transform_->is_convertible())
    storage_part = storage_->find_key_partition(machine, transform_->invert(restrictions));

  std::unique_ptr<Partition> store_part = nullptr;
  if (nullptr == storage_part || (!transform_->identity() && !storage_part->is_convertible())) {
    auto part_mgr     = Runtime::get_runtime()->partition_manager();
    auto launch_shape = part_mgr->compute_launch_shape(machine, restrictions, extents_);
    if (launch_shape.empty())
      store_part = create_no_partition();
    else {
      auto tile_shape = part_mgr->compute_tile_shape(extents_, launch_shape);
      store_part      = create_tiling(std::move(tile_shape), std::move(launch_shape));
    }
  } else
    store_part = transform_->convert(storage_part);
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(store_part != nullptr); }
  return store_part;
}

bool LogicalStore::has_key_partition(const mapping::detail::Machine& machine,
                                     const Restrictions& restrictions) const
{
  uint32_t new_num_pieces = machine.count();
  if (key_partition_ != nullptr && new_num_pieces == num_pieces_ &&
      key_partition_->satisfies_restrictions(restrictions))
    return true;
  else
    return transform_->is_convertible() &&
           storage_->find_key_partition(machine, transform_->invert(restrictions)) != nullptr;
}

void LogicalStore::set_key_partition(const mapping::detail::Machine& machine,
                                     const Partition* partition)
{
  num_pieces_ = machine.count();
  key_partition_.reset(partition->clone().release());
  auto inverted = transform_->invert(partition);
  storage_->set_key_partition(machine, std::move(inverted));
}

void LogicalStore::reset_key_partition()
{
  // Need to flush scheduling window to make this effective
  Runtime::get_runtime()->flush_scheduling_window();
  key_partition_ = nullptr;
  storage_->reset_key_partition();
}

std::shared_ptr<LogicalStorePartition> LogicalStore::create_partition(
  std::shared_ptr<Partition> partition, std::optional<bool> complete)
{
  if (unbound()) { throw std::invalid_argument("Unbound store cannot be manually partitioned"); }
  auto storage_partition =
    storage_->create_partition(transform_->invert(partition.get()), complete);
  return std::make_shared<LogicalStorePartition>(
    std::move(partition), std::move(storage_partition), shared_from_this());
}

void LogicalStore::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(has_scalar_storage());
  buffer.pack<bool>(unbound());
  buffer.pack<int32_t>(dim());
  type()->pack(buffer);
  transform_->pack(buffer);
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg(const Variable* variable,
                                                          const Strategy& strategy,
                                                          const Domain& launch_domain,
                                                          Legion::PrivilegeMode privilege,
                                                          int32_t redop)
{
  if (has_scalar_storage()) {
    if (!launch_domain.is_valid() && LEGION_REDUCE == privilege) { privilege = LEGION_READ_WRITE; }
    auto read_only   = privilege == LEGION_READ_ONLY;
    auto has_storage = get_future().valid() && privilege != LEGION_REDUCE;
    return std::make_unique<FutureStoreArg>(this, read_only, has_storage, redop);
  } else if (unbound()) {
    return std::make_unique<OutputRegionArg>(this, strategy.find_field_space(variable));
  } else {
    auto partition       = strategy[variable];
    auto store_partition = create_partition(partition);
    auto proj_info       = store_partition->create_projection_info(launch_domain);
    proj_info->is_key    = strategy.is_key_partition(variable);
    proj_info->redop     = redop;

    if (privilege == LEGION_REDUCE && store_partition->is_disjoint_for(launch_domain)) {
      privilege = LEGION_READ_WRITE;
    }
    if (privilege == LEGION_WRITE_ONLY || privilege == LEGION_READ_WRITE) {
      set_key_partition(variable->operation()->machine(), partition.get());
    }
    return std::make_unique<RegionFieldArg>(this, privilege, std::move(proj_info));
  }
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg_for_fixup(const Domain& launch_domain,
                                                                    Legion::PrivilegeMode privilege)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(key_partition_ != nullptr); }
  auto store_partition = create_partition(key_partition_);
  auto proj_info       = store_partition->create_projection_info(launch_domain);
  return std::make_unique<RegionFieldArg>(this, privilege, std::move(proj_info));
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
LogicalStorePartition::LogicalStorePartition(std::shared_ptr<Partition> partition,
                                             std::shared_ptr<StoragePartition> storage_partition,
                                             std::shared_ptr<LogicalStore> store)
  : partition_(std::move(partition)),
    storage_partition_(std::move(storage_partition)),
    store_(std::move(store))
{
}

// FIXME pass projection functor
std::unique_ptr<ProjectionInfo> LogicalStorePartition::create_projection_info(
  const Domain& launch_domain, std::optional<proj::SymbolicFunctor> proj_fn)
{
  if (store_->has_scalar_storage()) return std::make_unique<ProjectionInfo>();

  if (!partition_->has_launch_domain()) { return std::make_unique<ProjectionInfo>(); }

  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  auto legion_partition = storage_partition_->get_legion_partition();
  auto proj_id =
    launch_domain.is_valid() ? store_->compute_projection(launch_domain.dim, proj_fn) : 0;
  return std::make_unique<ProjectionInfo>(legion_partition, proj_id);
}

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return storage_partition_->is_disjoint_for(launch_domain);
}

const Shape& LogicalStorePartition::color_shape() const { return partition_->color_shape(); }

}  // namespace legate::detail
