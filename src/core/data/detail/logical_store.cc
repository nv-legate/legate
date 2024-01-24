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

#include "core/data/detail/physical_store.h"
#include "core/data/detail/transform.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/operation.h"
#include "core/operation/detail/store_projection.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/tuple.h"

#include "legate_defines.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::detail::Storage
////////////////////////////////////////////////////

Storage::Storage(uint32_t dim, InternalSharedPtr<Type> type)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    unbound_{true},
    shape_{make_internal_shared<Shape>(dim)},
    type_{std::move(type)},
    offsets_{legate::full(dim, uint64_t{0})}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(const InternalSharedPtr<Shape>& shape,
                 InternalSharedPtr<Type> type,
                 bool optimize_scalar)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    shape_{shape},
    type_{std::move(type)},
    offsets_{legate::full(dim(), uint64_t{0})}
{
  // We should not blindly check the shape volume as it would flush the scheduling window
  if (optimize_scalar && shape_->ready() && volume() == 1) {
    kind_ = Kind::FUTURE;
  }
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(const InternalSharedPtr<Shape>& shape,
                 InternalSharedPtr<Type> type,
                 const Legion::Future& future)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    shape_{shape},
    type_{std::move(type)},
    kind_{Kind::FUTURE},
    future_{std::make_unique<Legion::Future>(future)},
    offsets_{legate::full(dim(), uint64_t{0})}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(tuple<uint64_t>&& extents,
                 InternalSharedPtr<Type> type,
                 InternalSharedPtr<StoragePartition> parent,
                 tuple<uint64_t>&& color,
                 tuple<uint64_t>&& offsets)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    type_{std::move(type)},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    color_{std::move(color)},
    offsets_{std::move(offsets)}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::~Storage()
{
  if (!Runtime::get_runtime()->initialized()) {
    // FIXME: Leak the Future handle if the runtime has already shut down, as there's no hope that
    // this would be collected by the Legion runtime
    static_cast<void>(future_.release());
  }
}

const tuple<uint64_t>& Storage::offsets() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!unbound_);
  }
  return offsets_;
}

bool Storage::overlaps(const InternalSharedPtr<Storage>& other) const
{
  const auto* lhs = this;
  const auto* rhs = other.get();

  if (lhs == rhs) {
    return true;
  }

  if (lhs->get_root() != rhs->get_root()) {
    return false;
  }

  if (lhs->volume() == 0 || rhs->volume() == 0) {
    return false;
  }

  auto& lexts = lhs->extents();
  auto& rexts = rhs->extents();

  for (uint32_t idx = 0; idx < dim(); ++idx) {
    auto lext = lexts[idx];
    auto rext = rexts[idx];
    auto loff = lhs->offsets_[idx];
    auto roff = rhs->offsets_[idx];

    if (loff <= roff ? roff < loff + lext : loff < roff + rext) {
      continue;
    }
    return false;
  }
  return true;
}

InternalSharedPtr<Storage> Storage::slice(tuple<uint64_t> tile_shape,
                                          const tuple<uint64_t>& offsets)
{
  if (Kind::FUTURE == kind_) {
    return shared_from_this();
  }

  const auto root  = get_root();
  const auto shape = root->extents();

  const auto can_tile_completely =
    (shape % tile_shape).sum() == 0 && (offsets % tile_shape).sum() == 0 &&
    Runtime::get_runtime()->partition_manager()->use_complete_tiling(shape, tile_shape);

  tuple<uint64_t> color_shape, color;
  tuple<int64_t> signed_offsets;
  if (can_tile_completely) {
    color_shape    = shape / tile_shape;
    color          = offsets / tile_shape;
    signed_offsets = legate::full<int64_t>(shape.size(), 0);
  } else {
    color_shape    = legate::full<uint64_t>(shape.size(), 1);
    color          = legate::full<uint64_t>(shape.size(), 0);
    signed_offsets = apply([](size_t v) { return static_cast<int64_t>(v); }, offsets);
  }

  auto tiling =
    create_tiling(std::move(tile_shape), std::move(color_shape), std::move(signed_offsets));
  auto storage_partition = root->create_partition(std::move(tiling), can_tile_completely);
  return storage_partition->get_child_storage(std::move(color));
}

InternalSharedPtr<const Storage> Storage::get_root() const
{
  return parent_ ? parent_->get_root() : shared_from_this();
}

InternalSharedPtr<Storage> Storage::get_root()
{
  return parent_ ? parent_->get_root() : shared_from_this();
}

InternalSharedPtr<LogicalRegionField> Storage::get_region_field()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(Kind::REGION_FIELD == kind_);
  }
  if (region_field_ != nullptr) {
    return region_field_;
  }

  if (nullptr == parent_) {
    region_field_ = Runtime::get_runtime()->create_region_field(shape_, type_->size());
    if (destroyed_out_of_order_) {
      region_field_->allow_out_of_order_destruction();
    }
  } else {
    region_field_ = parent_->get_child_data(color_);
  }
  return region_field_;
}

Legion::Future Storage::get_future() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(Kind::FUTURE == kind_);
  }
  return future_ != nullptr ? *future_ : Legion::Future{};
}

void Storage::set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field)
{
  assert(unbound_ && region_field_ == nullptr);
  assert(parent_ == nullptr);

  unbound_      = false;
  region_field_ = std::move(region_field);
  if (destroyed_out_of_order_) {
    region_field_->allow_out_of_order_destruction();
  }
}

void Storage::set_future(Legion::Future future)
{
  future_ = std::make_unique<Legion::Future>(std::move(future));
}

RegionField Storage::map()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(Kind::REGION_FIELD == kind_);
  }
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
  if (parent_) {
    get_root()->allow_out_of_order_destruction();
  } else if (!destroyed_out_of_order_) {
    destroyed_out_of_order_ = true;
    if (region_field_) {
      region_field_->allow_out_of_order_destruction();
    }
  }
}

Restrictions Storage::compute_restrictions() const
{
  return legate::full<Restriction>(dim(), Restriction::ALLOW);
}

Partition* Storage::find_key_partition(const mapping::detail::Machine& machine,
                                       const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count();

  if ((num_pieces_ == new_num_pieces) && key_partition_ &&
      key_partition_->satisfies_restrictions(restrictions)) {
    return key_partition_.get();
  }
  if (parent_) {
    return parent_->find_key_partition(machine, restrictions);
  }
  return {};
}

void Storage::set_key_partition(const mapping::detail::Machine& machine,
                                std::unique_ptr<Partition>&& key_partition)
{
  num_pieces_    = machine.count();
  key_partition_ = std::move(key_partition);
}

void Storage::reset_key_partition() noexcept { key_partition_.reset(); }

InternalSharedPtr<StoragePartition> Storage::create_partition(
  InternalSharedPtr<Partition> partition, std::optional<bool> complete)
{
  if (!complete.has_value()) {
    complete = partition->is_complete_for(this);
  }
  return make_internal_shared<StoragePartition>(
    shared_from_this(), std::move(partition), complete.value());
}

std::string Storage::to_string() const
{
  std::stringstream ss;

  ss << "Storage(" << storage_id_ << ") {" << shape_->to_string()
     << ", kind: " << (kind_ == Kind::REGION_FIELD ? "Region" : "Future")
     << ", type: " << type_->to_string() << ", level: " << level_ << "}";

  return std::move(ss).str();
}

////////////////////////////////////////////////////
// legate::detail::StoragePartition
////////////////////////////////////////////////////

InternalSharedPtr<const Storage> StoragePartition::get_root() const { return parent_->get_root(); }

InternalSharedPtr<Storage> StoragePartition::get_root() { return parent_->get_root(); }

InternalSharedPtr<Storage> StoragePartition::get_child_storage(tuple<uint64_t> color)
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw std::runtime_error{"Sub-storage is implemented only for tiling"};
  }

  auto tiling        = static_cast<Tiling*>(partition_.get());
  auto child_extents = tiling->get_child_extents(parent_->extents(), color);
  auto child_offsets = tiling->get_child_offsets(color);
  return make_internal_shared<Storage>(std::move(child_extents),
                                       parent_->type(),
                                       shared_from_this(),
                                       std::move(color),
                                       std::move(child_offsets));
}

InternalSharedPtr<LogicalRegionField> StoragePartition::get_child_data(const tuple<uint64_t>& color)
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw std::runtime_error{"Sub-storage is implemented only for tiling"};
  }

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

namespace {

void assert_fixed_storage_size(const InternalSharedPtr<Storage>& storage)
{
  if (storage->type()->variable_size()) {
    throw std::invalid_argument{"Store cannot be created with variable size type " +
                                storage->type()->to_string()};
  }
}

[[nodiscard]] InternalSharedPtr<TransformStack> stack(
  const InternalSharedPtr<TransformStack>& parent, std::unique_ptr<StoreTransform>&& transform)
{
  return make_internal_shared<TransformStack>(std::move(transform), parent);
}

}  // namespace

LogicalStore::LogicalStore(InternalSharedPtr<Storage>&& storage)
  : store_id_{Runtime::get_runtime()->get_unique_store_id()},
    shape_{storage->shape()},
    storage_{std::move(storage)},
    transform_{make_internal_shared<TransformStack>()}
{
  assert_fixed_storage_size(storage_);
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(transform_ != nullptr);

    log_legate().debug() << "Create " << to_string();
  }
}

LogicalStore::LogicalStore(tuple<uint64_t>&& extents,
                           InternalSharedPtr<Storage> storage,
                           InternalSharedPtr<TransformStack>&& transform)
  : store_id_{Runtime::get_runtime()->get_unique_store_id()},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    storage_{std::move(storage)},
    transform_{std::move(transform)}
{
  assert_fixed_storage_size(storage_);
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(transform_ != nullptr);

    log_legate().debug() << "Create " << to_string();
  }
}

void LogicalStore::set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!has_scalar_storage());
    // The shape object of this store must be an alias to that of the storage
    assert(storage_->shape() == shape());
  }
  // this call updates the shape for both the storage and the store
  storage_->set_region_field(std::move(region_field));
}

void LogicalStore::set_future(Legion::Future future)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(has_scalar_storage());
  }
  storage_->set_future(std::move(future));
}

InternalSharedPtr<LogicalStore> LogicalStore::promote(int32_t extra_dim, size_t dim_size)
{
  if (extra_dim < 0 || extra_dim > static_cast<int32_t>(dim())) {
    throw std::invalid_argument{"Invalid promotion on dimension " + std::to_string(extra_dim) +
                                " for a " + std::to_string(dim()) + "-D store"};
  }

  auto new_extents = extents().insert(extra_dim, dim_size);
  auto transform   = stack(transform_, std::make_unique<Promote>(extra_dim, dim_size));
  return make_internal_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::project(int32_t d, int64_t index)
{
  auto old_extents = extents();

  if (d < 0 || d >= static_cast<int32_t>(dim())) {
    throw std::invalid_argument{"Invalid projection on dimension " + std::to_string(d) + " for a " +
                                std::to_string(dim()) + "-D store"};
  }

  if (index < 0 || index >= static_cast<int64_t>(old_extents[d])) {
    throw std::invalid_argument{"Projection index " + std::to_string(index) +
                                " is out of bounds [0, " + std::to_string(old_extents[d]) + ")"};
  }

  auto new_extents = old_extents.remove(d);
  auto transform   = stack(transform_, std::make_unique<Project>(d, index));
  auto substorage =
    volume() == 0
      ? storage_
      : storage_->slice(transform->invert_extents(new_extents),
                        transform->invert_point(legate::full<uint64_t>(new_extents.size(), 0)));
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), std::move(substorage), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::slice(const InternalSharedPtr<LogicalStore>& self,
                                                    int32_t dim,
                                                    Slice slice)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(self.get() == this);
  }
  if (dim < 0 || dim >= static_cast<int32_t>(this->dim())) {
    throw std::invalid_argument{"Invalid slicing of dimension " + std::to_string(dim) + " for a " +
                                std::to_string(this->dim()) + "-D store"};
  }

  auto sanitize_slice = [](const Slice& san_slice, int64_t extent) {
    int64_t start = san_slice.start.value_or(0);
    int64_t stop  = san_slice.stop.value_or(extent);

    if (start < 0) {
      start += extent;
    }
    if (stop < 0) {
      stop += extent;
    }

    return std::make_pair<size_t, size_t>(std::max(int64_t{0}, start), std::max(int64_t{0}, stop));
  };

  auto exts          = extents();
  auto [start, stop] = sanitize_slice(slice, static_cast<int64_t>(exts[dim]));

  if (start < stop && (start >= exts[dim] || stop > exts[dim])) {
    throw std::invalid_argument{"Out-of-bounds slicing on dimension " +
                                std::to_string(this->dim()) + " for a store of shape " +
                                extents().to_string()};
  }

  exts[dim] = (start < stop) ? (stop - start) : 0;

  if (exts[dim] == extents()[dim]) {
    return self;
  }

  if (0 == exts[dim]) {
    return Runtime::get_runtime()->create_store(make_internal_shared<Shape>(std::move(exts)),
                                                type());
  }

  auto transform =
    (start == 0) ? transform_ : stack(transform_, std::make_unique<Shift>(dim, -start));
  auto substorage =
    volume() == 0
      ? storage_
      : storage_->slice(transform->invert_extents(exts),
                        transform->invert_point(legate::full<uint64_t>(exts.size(), 0)));
  return make_internal_shared<LogicalStore>(
    std::move(exts), std::move(substorage), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::transpose(std::vector<int32_t> axes)
{
  if (axes.size() != dim()) {
    throw std::invalid_argument{"Dimension Mismatch: expected " + std::to_string(dim()) +
                                " axes, but got " + std::to_string(axes.size())};
  }

  if (axes.size() != std::set<int32_t>{axes.begin(), axes.end()}.size()) {
    throw std::invalid_argument{"Duplicate axes found"};
  }

  for (auto&& ax_i : axes) {
    if (ax_i < 0 || ax_i >= static_cast<int32_t>(dim())) {
      throw std::invalid_argument{"Invalid axis " + std::to_string(ax_i) + " for a " +
                                  std::to_string(dim()) + "-D store"};
    }
  }

  auto old_extents = extents();
  auto new_extents = tuple<uint64_t>{};

  new_extents.reserve(axes.size());
  for (auto&& ax_i : axes) {
    new_extents.append_inplace(old_extents[ax_i]);
  }

  auto transform = stack(transform_, std::make_unique<Transpose>(std::move(axes)));
  return make_internal_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::delinearize(int32_t dim, std::vector<uint64_t> sizes)
{
  if (dim < 0 || dim >= static_cast<int32_t>(this->dim())) {
    throw std::invalid_argument{"Invalid delinearization on dimension " + std::to_string(dim) +
                                " for a " + std::to_string(this->dim()) + "-D store"};
  }

  auto old_extents = extents();

  auto delinearizable = [](auto&& to_match, const auto& new_dim_extents) {
    auto begin = new_dim_extents.begin();
    auto end   = new_dim_extents.end();
    return std::reduce(begin, end, size_t{1}, std::multiplies<>{}) == to_match &&
           // overflow check
           std::all_of(begin, end, [&to_match](auto size) { return size <= to_match; });
  };

  if (!delinearizable(old_extents[dim], sizes)) {
    throw std::invalid_argument{"Dimension of size " + std::to_string(old_extents[dim]) +
                                " cannot be delinearized into " +
                                tuple<uint64_t>{sizes}.to_string()};
  }

  auto new_extents = tuple<uint64_t>{};

  new_extents.reserve(dim);
  for (int i = 0; i < dim; i++) {
    new_extents.append_inplace(old_extents[i]);
  }
  for (auto&& size : sizes) {
    new_extents.append_inplace(size);
  }
  for (auto i = static_cast<uint32_t>(dim + 1); i < old_extents.size(); i++) {
    new_extents.append_inplace(old_extents[i]);
  }

  auto transform = stack(transform_, std::make_unique<Delinearize>(dim, std::move(sizes)));
  return make_internal_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::partition_by_tiling(
  const InternalSharedPtr<LogicalStore>& self, tuple<uint64_t> tile_shape)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(self.get() == this);
  }
  if (tile_shape.size() != dim()) {
    throw std::invalid_argument{"Incompatible tile shape: expected a " +
                                std::to_string(extents().size()) + "-tuple, got a " +
                                std::to_string(tile_shape.size()) + "-tuple"};
  }
  if (tile_shape.volume() == 0) {
    throw std::invalid_argument{"Tile shape must have a volume greater than 0"};
  }
  auto color_shape = apply([](auto c, auto t) { return (c + t - 1) / t; }, extents(), tile_shape);
  auto partition   = create_tiling(std::move(tile_shape), std::move(color_shape));
  return create_partition(self, std::move(partition), true);
}

InternalSharedPtr<PhysicalStore> LogicalStore::get_physical_store()
{
  if (unbound()) {
    throw std::invalid_argument{"Unbound store cannot be inlined mapped"};
  }
  if (mapped_) {
    return mapped_;
  }
  if (storage_->kind() == Storage::Kind::FUTURE) {
    // TODO(wonchanl): future wrappers from inline mappings are read-only for now
    auto domain = to_domain(storage_->shape()->extents());
    auto future = FutureWrapper{true, type()->size(), domain, storage_->get_future()};
    // Physical stores for future-backed stores shouldn't be cached, as they are not automatically
    // remapped to reflect changes by the runtime.
    return make_internal_shared<PhysicalStore>(dim(), type(), -1, std::move(future), transform_);
  }

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(storage_->kind() == Storage::Kind::REGION_FIELD);
  }
  auto region_field = storage_->map();
  mapped_ =
    make_internal_shared<PhysicalStore>(dim(), type(), -1, std::move(region_field), transform_);
  return mapped_;
}

void LogicalStore::detach()
{
  if (transformed()) {
    throw std::invalid_argument{"Manual detach must be called on the root store"};
  }
  if (has_scalar_storage() || unbound()) {
    throw std::invalid_argument{"Only stores created with share=true can be manually detached"};
  }
  get_region_field()->detach();
}

void LogicalStore::allow_out_of_order_destruction()
{
  if (Runtime::get_runtime()->consensus_match_required()) {
    storage_->allow_out_of_order_destruction();
  }
}

Restrictions LogicalStore::compute_restrictions(bool is_output) const
{
  return transform_->convert(storage_->compute_restrictions(), is_output);
}

Legion::ProjectionID LogicalStore::compute_projection(
  uint32_t launch_ndim, const std::optional<SymbolicPoint>& projection) const
{
  if (projection) {
    assert(!transformed());
    return Runtime::get_runtime()->get_projection(launch_ndim, *projection);
  }

  if (transform_->identity()) {
    if (launch_ndim != dim()) {
      return Runtime::get_runtime()->get_delinearizing_projection();
    }
    return 0;
  }

  const auto ndim = dim();
  // TODO(wonchanl): We can't currently mix affine projections with delinearizing projections
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(ndim == launch_ndim);
  }
  return Runtime::get_runtime()->get_projection(
    ndim, transform_->invert(proj::create_symbolic_point(ndim)));
}

InternalSharedPtr<Partition> LogicalStore::find_or_create_key_partition(
  const mapping::detail::Machine& machine, const Restrictions& restrictions)
{
  const auto new_num_pieces = machine.count();

  if ((num_pieces_ == new_num_pieces) && key_partition_ &&
      key_partition_->satisfies_restrictions(restrictions)) {
    return key_partition_;
  }

  if (has_scalar_storage() || dim() == 0 || volume() == 0) {
    return create_no_partition();
  }

  Partition* storage_part{};

  if (transform_->is_convertible()) {
    storage_part = storage_->find_key_partition(machine, transform_->invert(restrictions));
  }

  std::unique_ptr<Partition> store_part{};
  if (nullptr == storage_part || (!transform_->identity() && !storage_part->is_convertible())) {
    auto& exts        = extents();
    auto part_mgr     = Runtime::get_runtime()->partition_manager();
    auto launch_shape = part_mgr->compute_launch_shape(machine, restrictions, exts);

    if (launch_shape.empty()) {
      store_part = create_no_partition();
    } else {
      auto tile_shape = part_mgr->compute_tile_shape(exts, launch_shape);

      store_part = create_tiling(std::move(tile_shape), std::move(launch_shape));
    }
  } else {
    store_part = transform_->convert(storage_part);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(store_part);
    }
  }
  return store_part;
}

bool LogicalStore::has_key_partition(const mapping::detail::Machine& machine,
                                     const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count();

  if ((new_num_pieces == num_pieces_) && key_partition_ &&
      key_partition_->satisfies_restrictions(restrictions)) {
    return true;
  }
  return transform_->is_convertible() &&
         storage_->find_key_partition(machine, transform_->invert(restrictions)) != nullptr;
}

void LogicalStore::set_key_partition(const mapping::detail::Machine& machine,
                                     const Partition* partition)
{
  num_pieces_ = machine.count();
  key_partition_.reset(partition->clone().release());
  storage_->set_key_partition(machine, transform_->invert(partition));
}

void LogicalStore::reset_key_partition()
{
  // Need to flush scheduling window to make this effective
  Runtime::get_runtime()->flush_scheduling_window();
  key_partition_.reset();
  storage_->reset_key_partition();
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::create_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
#if LegateDefined(LEGATE_USE_DEBUG)
  assert(self.get() == this);
#endif
  if (unbound()) {
    throw std::invalid_argument{"Unbound store cannot be manually partitioned"};
  }
  auto storage_partition =
    storage_->create_partition(transform_->invert(partition.get()), std::move(complete));
  return make_internal_shared<LogicalStorePartition>(
    std::move(partition), std::move(storage_partition), self);
}

void LogicalStore::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(has_scalar_storage());
  buffer.pack<bool>(unbound());
  buffer.pack<uint32_t>(dim());
  type()->pack(buffer);
  transform_->pack(buffer);
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  int64_t redop)
{
#if LegateDefined(LEGATE_USE_DEBUG)
  assert(self.get() == this);
#endif
  if (has_scalar_storage()) {
    if (!launch_domain.is_valid() && LEGION_REDUCE == privilege) {
      privilege = LEGION_READ_WRITE;
    }
    auto read_only   = privilege == LEGION_READ_ONLY;
    auto has_storage = get_future().valid() && privilege != LEGION_REDUCE;

    return std::make_unique<FutureStoreArg>(this, read_only, has_storage, redop);
  }

  if (unbound()) {
    return std::make_unique<OutputRegionArg>(this, strategy.find_field_space(variable));
  }

  auto partition       = strategy[variable];
  auto store_partition = create_partition(self, partition);
  auto store_proj      = store_partition->create_store_projection(launch_domain, projection);
  store_proj->is_key   = strategy.is_key_partition(variable);
  store_proj->redop    = static_cast<Legion::ReductionOpID>(redop);

  if (privilege == LEGION_REDUCE && store_partition->is_disjoint_for(launch_domain)) {
    privilege = LEGION_READ_WRITE;
  }
  if (privilege == LEGION_WRITE_ONLY || privilege == LEGION_READ_WRITE) {
    set_key_partition(variable->operation()->machine(), partition.get());
  }

  return std::make_unique<RegionFieldArg>(this, privilege, std::move(store_proj));
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg_for_fixup(
  const InternalSharedPtr<LogicalStore>& self,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege)
{
#if LegateDefined(LEGATE_USE_DEBUG)
  assert(self.get() == this);
#endif
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(self->key_partition_ != nullptr);
  }
  auto store_partition = create_partition(self, self->key_partition_);
  auto store_proj      = store_partition->create_store_projection(launch_domain);
  return std::make_unique<RegionFieldArg>(this, privilege, std::move(store_proj));
}

std::string LogicalStore::to_string() const
{
  std::stringstream ss;

  ss << "Store(" << store_id_ << ") {shape: ";
  if (unbound()) {
    ss << "(unbound)";
  } else {
    ss << extents();
  }
  if (!transform_->identity()) {
    ss << ", transform: " << *transform_;
  }
  ss << ", storage: " << storage_->id() << "}";
  return std::move(ss).str();
}

////////////////////////////////////////////////////
// legate::detail::LogicalStorePartition
////////////////////////////////////////////////////

InternalSharedPtr<LogicalStore> LogicalStorePartition::get_child_store(
  const tuple<uint64_t>& color) const
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw std::runtime_error{"Child stores can be retrieved only from tile partitions"};
  }
  const auto* tiling = static_cast<const Tiling*>(partition_.get());

  auto transform      = store_->transform();
  auto inverted_color = transform->invert_color(color);
  auto child_storage  = storage_partition_->get_child_storage(inverted_color);

  auto child_extents = tiling->get_child_extents(store_->extents(), inverted_color);
  auto child_offsets = tiling->get_child_offsets(inverted_color);

  for (uint32_t dim = 0; dim < child_offsets.size(); ++dim) {
    if (child_offsets[dim] == 0) {
      continue;
    }
    transform = make_internal_shared<TransformStack>(
      std::make_unique<Shift>(dim, -child_offsets[dim]), std::move(transform));
  }

  return make_internal_shared<LogicalStore>(
    std::move(child_extents), std::move(child_storage), std::move(transform));
}

std::unique_ptr<StoreProjection> LogicalStorePartition::create_store_projection(
  const Domain& launch_domain, const std::optional<SymbolicPoint>& projection)
{
  if (store_->has_scalar_storage()) {
    return std::make_unique<StoreProjection>();
  }

  if (!partition_->has_launch_domain()) {
    return std::make_unique<StoreProjection>();
  }

  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  auto legion_partition = storage_partition_->get_legion_partition();
  auto proj_id          = launch_domain.is_valid() ? store_->compute_projection(
                                              static_cast<uint32_t>(launch_domain.dim), projection)
                                                   : 0;
  return std::make_unique<StoreProjection>(std::move(legion_partition), proj_id);
}

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return storage_partition_->is_disjoint_for(launch_domain);
}

const tuple<uint64_t>& LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

}  // namespace legate::detail
