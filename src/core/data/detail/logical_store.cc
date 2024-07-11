/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "core/operation/projection.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/enumerate.h"
#include "core/utilities/detail/tuple.h"

#include "legate_defines.h"

#include <algorithm>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::detail::Storage
////////////////////////////////////////////////////

Storage::Storage(InternalSharedPtr<Shape> shape,
                 InternalSharedPtr<Type> type,
                 bool optimize_scalar,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    unbound_{shape->unbound()},
    shape_{std::move(shape)},
    type_{std::move(type)},
    offsets_{legate::full(dim(), uint64_t{0})},
    provenance_{std::move(provenance)}
{
  // We should not blindly check the shape volume as it would flush the scheduling window
  if (optimize_scalar) {
    if (shape_->unbound()) {
      kind_ = Kind::FUTURE_MAP;
    } else if (shape_->ready() && volume() == 1) {
      kind_ = Kind::FUTURE;
    }
  }
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(InternalSharedPtr<Shape> shape,
                 InternalSharedPtr<Type> type,
                 Legion::Future future,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    shape_{std::move(shape)},
    type_{std::move(type)},
    kind_{Kind::FUTURE},
    future_{std::move(future)},
    offsets_{legate::full(dim(), uint64_t{0})},
    provenance_{std::move(provenance)}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(tuple<std::uint64_t> extents,
                 InternalSharedPtr<Type> type,
                 InternalSharedPtr<StoragePartition> parent,
                 tuple<std::uint64_t> color,
                 tuple<std::uint64_t> offsets)
  : storage_id_{Runtime::get_runtime()->get_unique_storage_id()},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    type_{std::move(type)},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    color_{std::move(color)},
    offsets_{std::move(offsets)}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
Storage::~Storage()
{
  if (!Runtime::get_runtime()->initialized()) {
    if (future_.has_value() && future_->exists()) {
      // FIXME: Leak the Future handle if the runtime has already shut down, as there's no hope that
      // this would be collected by the Legion runtime
      static_cast<void>(std::make_unique<Legion::Future>(*std::move(future_)).release());
    }
    if (future_map_.has_value()) {
      static_cast<void>(std::make_unique<Legion::FutureMap>(*std::move(future_map_)).release());
    }
  }
}
// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

const tuple<std::uint64_t>& Storage::offsets() const
{
  LEGATE_ASSERT(!unbound_);
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

  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
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

InternalSharedPtr<Storage> Storage::slice(tuple<std::uint64_t> tile_shape,
                                          const tuple<std::uint64_t>& offsets)
{
  if (Kind::FUTURE == kind_) {
    return shared_from_this();
  }

  const auto root   = get_root();
  const auto& shape = root->extents();
  const auto can_tile_completely =
    (shape % tile_shape).sum() == 0 && (offsets % tile_shape).sum() == 0 &&
    Runtime::get_runtime()->partition_manager()->use_complete_tiling(shape, tile_shape);

  tuple<std::uint64_t> color_shape, color;
  tuple<std::int64_t> signed_offsets;
  if (can_tile_completely) {
    color_shape    = shape / tile_shape;
    color          = offsets / tile_shape;
    signed_offsets = legate::full<std::int64_t>(shape.size(), 0);
  } else {
    color_shape    = legate::full<std::uint64_t>(shape.size(), 1);
    color          = legate::full<std::uint64_t>(shape.size(), 0);
    signed_offsets = apply([](auto&& v) { return static_cast<std::int64_t>(v); }, offsets);
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

const InternalSharedPtr<LogicalRegionField>& Storage::get_region_field()
{
  LEGATE_CHECK(kind_ == Kind::REGION_FIELD);
  if (region_field_) {
    return region_field_;
  }

  if (nullptr == parent_) {
    region_field_ = Runtime::get_runtime()->create_region_field(shape_, type_->size());
    if (destroyed_out_of_order_) {
      region_field_->allow_out_of_order_destruction();
    }
    Runtime::get_runtime()->attach_alloc_info(region_field_, provenance());
  } else {
    region_field_ = parent_->get_child_data(color_);
  }
  return region_field_;
}

Legion::Future Storage::get_future() const
{
  if (kind_ == Kind::FUTURE) {
    return future_.value_or(Legion::Future{});
  }

  LEGATE_ASSERT(kind_ == Kind::FUTURE_MAP);

  // this future map must always exist, otherwise something bad has happened
  auto future_map = get_future_map();
  return future_map.get_future(future_map.get_future_map_domain().lo());
}

Legion::FutureMap Storage::get_future_map() const
{
  LEGATE_CHECK(kind_ == Kind::FUTURE_MAP);

  auto&& future_map = [&] {
    // this future map must always exist, otherwise something bad has happened
    try {
      return future_map_.value();
    } catch (const std::bad_optional_access&) {
      LEGATE_ABORT("Future map must have existed");
    }
    LEGATE_UNREACHABLE();
    return Legion::FutureMap{};
  }();

  return future_map;
}

std::variant<Legion::Future, Legion::FutureMap> Storage::get_future_or_future_map(
  const Domain& launch_domain) const
{
  LEGATE_ASSERT(kind_ == Kind::FUTURE_MAP);

  // this future map must always exist, otherwise something bad has happened
  auto future_map        = get_future_map();
  auto future_map_domain = future_map.get_future_map_domain();

  if (!launch_domain.is_valid()) {
    return future_map.get_future(future_map_domain.lo());
  }
  if (launch_domain == future_map_domain) {
    return future_map;
  }
  if (launch_domain.get_volume() != future_map_domain.get_volume()) {
    return future_map.get_future(future_map_domain.lo());
  }
  return Runtime::get_runtime()->reshape_future_map(future_map, launch_domain);
}

void Storage::set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field)
{
  LEGATE_CHECK(unbound_ && region_field_ == nullptr);
  LEGATE_CHECK(parent_ == nullptr);

  unbound_      = false;
  region_field_ = std::move(region_field);
  if (destroyed_out_of_order_) {
    region_field_->allow_out_of_order_destruction();
  }
  Runtime::get_runtime()->attach_alloc_info(region_field_, provenance());
}

void Storage::set_future(Legion::Future future, std::size_t scalar_offset)
{
  scalar_offset_ = scalar_offset;
  future_        = std::move(future);
  // If we're here, that means that this was a replicated future that gets updated via reductions,
  // so we reset the stale future map and update the kind
  if (kind() == Storage::Kind::FUTURE_MAP) {
    // TODO(wonchanl): true future map-backed stores aren't exposed to the user yet
    // so if it wasn't replicated, something bad must have happened
    LEGATE_CHECK(replicated_);
    kind_       = Storage::Kind::FUTURE;
    replicated_ = false;
    future_map_.reset();
  }
}

void Storage::set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset)
{
  scalar_offset_ = scalar_offset;
  future_map_    = std::move(future_map);
  // If this was originally a future-backed storage, it means this storage is now backed by a future
  // map with futures having the same value
  if (kind() == Storage::Kind::FUTURE) {
    kind_       = Storage::Kind::FUTURE_MAP;
    replicated_ = true;
    future_.reset();
  }
}

RegionField Storage::map()
{
  LEGATE_ASSERT(Kind::REGION_FIELD == kind_);
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
  const auto kind_str = [&] {
    switch (kind_) {
      case Kind::REGION_FIELD: return "Region";
      case Kind::FUTURE: return "Future";
      case Kind::FUTURE_MAP: return "Future map";
    }
    LEGATE_UNREACHABLE();
  }();

  return fmt::format("Storage({}) {{{}, kind: {}, type: {}, level: {}}}",
                     storage_id_,
                     *shape_,
                     kind_str,
                     *type_,
                     level_);
}

////////////////////////////////////////////////////
// legate::detail::StoragePartition
////////////////////////////////////////////////////

InternalSharedPtr<const Storage> StoragePartition::get_root() const { return parent_->get_root(); }

InternalSharedPtr<Storage> StoragePartition::get_root() { return parent_->get_root(); }

InternalSharedPtr<Storage> StoragePartition::get_child_storage(tuple<std::uint64_t> color)
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

InternalSharedPtr<LogicalRegionField> StoragePartition::get_child_data(
  const tuple<std::uint64_t>& color)
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

void assert_fixed_storage_size(const Storage* storage)
{
  if (storage->type()->variable_size()) {
    throw std::invalid_argument{
      fmt::format("Store cannot be created with variable size type {}", *(storage->type()))};
  }
}

[[nodiscard]] InternalSharedPtr<TransformStack> stack(
  const InternalSharedPtr<TransformStack>& parent, std::unique_ptr<StoreTransform>&& transform)
{
  return make_internal_shared<TransformStack>(std::move(transform), parent);
}

}  // namespace

LogicalStore::LogicalStore(InternalSharedPtr<Storage> storage)
  : store_id_{Runtime::get_runtime()->get_unique_store_id()},
    shape_{storage->shape()},
    storage_{std::move(storage)},
    transform_{make_internal_shared<TransformStack>()}
{
  assert_fixed_storage_size(get_storage());
  LEGATE_ASSERT(transform_ != nullptr);
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

LogicalStore::LogicalStore(tuple<std::uint64_t> extents,
                           InternalSharedPtr<Storage> storage,
                           InternalSharedPtr<TransformStack> transform)
  : store_id_{Runtime::get_runtime()->get_unique_store_id()},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    storage_{std::move(storage)},
    transform_{std::move(transform)}
{
  assert_fixed_storage_size(get_storage());
  LEGATE_ASSERT(transform_ != nullptr);
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

void LogicalStore::set_region_field(InternalSharedPtr<LogicalRegionField> region_field)
{
  LEGATE_ASSERT(!has_scalar_storage());
  // The shape object of this store must be an alias to that of the storage
  LEGATE_ASSERT(get_storage()->shape() == shape());
  // this call updates the shape for both the storage and the store
  get_storage()->set_region_field(std::move(region_field));
}

void LogicalStore::set_future(Legion::Future future, std::size_t scalar_offset)
{
  LEGATE_ASSERT(has_scalar_storage());
  get_storage()->set_future(std::move(future), scalar_offset);
}

void LogicalStore::set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset)
{
  LEGATE_ASSERT(has_scalar_storage());
  get_storage()->set_future_map(std::move(future_map), scalar_offset);
}

InternalSharedPtr<LogicalStore> LogicalStore::promote(std::int32_t extra_dim, std::size_t dim_size)
{
  if (extra_dim < 0 || extra_dim > static_cast<std::int32_t>(dim())) {
    throw std::invalid_argument{
      fmt::format("Invalid promotion on dimension {} for a {}-D store", extra_dim, dim())};
  }

  auto new_extents = extents().insert(extra_dim, dim_size);
  auto transform   = stack(transform_, std::make_unique<Promote>(extra_dim, dim_size));
  return make_internal_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::project(std::int32_t d, std::int64_t index)
{
  auto&& old_extents = extents();

  if (d < 0 || d >= static_cast<std::int32_t>(dim())) {
    throw std::invalid_argument{
      fmt::format("Invalid projection on dimension {} for a {}-D store", d, dim())};
  }

  if (index < 0 || index >= static_cast<std::int64_t>(old_extents[d])) {
    throw std::invalid_argument{
      fmt::format("Projection index {} is out of bounds [0, {})", index, old_extents[d])};
  }

  auto new_extents = old_extents.remove(d);
  auto transform   = stack(transform_, std::make_unique<Project>(d, index));
  auto substorage =
    volume() == 0 ? storage_
                  : storage_->slice(
                      transform->invert_extents(new_extents),
                      transform->invert_point(legate::full<std::uint64_t>(new_extents.size(), 0)));
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), std::move(substorage), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::slice_(const InternalSharedPtr<LogicalStore>& self,
                                                     std::int32_t dim,
                                                     Slice slice)
{
  LEGATE_ASSERT(self.get() == this);
  if (dim < 0 || dim >= static_cast<std::int32_t>(this->dim())) {
    throw std::invalid_argument{
      fmt::format("Invalid slicing of dimension {} for a {}-D store", dim, this->dim())};
  }

  constexpr auto sanitize_slice = [](const Slice& san_slice, std::int64_t extent) {
    std::int64_t start = san_slice.start.value_or(0);
    std::int64_t stop  = san_slice.stop.value_or(extent);

    if (start < 0) {
      start += extent;
    }
    if (stop < 0) {
      stop += extent;
    }

    return std::make_pair<size_t, std::size_t>(std::max(int64_t{0}, start),
                                               std::max(int64_t{0}, stop));
  };

  auto exts          = extents();
  auto [start, stop] = sanitize_slice(slice, static_cast<std::int64_t>(exts[dim]));

  if (start < stop && (start >= exts[dim] || stop > exts[dim])) {
    throw std::invalid_argument{fmt::format(
      "Out-of-bounds slicing on dimension {} for a store of shape {}", this->dim(), extents())};
  }

  exts[dim] = (start < stop) ? (stop - start) : 0;

  if (exts[dim] == extents()[dim]) {
    return self;
  }

  if (0 == exts[dim]) {
    return Runtime::get_runtime()->create_store(
      make_internal_shared<Shape>(std::move(exts)), type(), false /*optimize_scalar*/);
  }

  auto transform =
    (start == 0) ? transform_ : stack(transform_, std::make_unique<Shift>(dim, -start));
  auto substorage =
    volume() == 0
      ? storage_
      : storage_->slice(transform->invert_extents(exts),
                        transform->invert_point(legate::full<std::uint64_t>(exts.size(), 0)));
  return make_internal_shared<LogicalStore>(
    std::move(exts), std::move(substorage), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::transpose(std::vector<std::int32_t> axes)
{
  if (axes.size() != dim()) {
    throw std::invalid_argument{
      fmt::format("Dimension Mismatch: expected {} axes, but got {}", dim(), axes.size())};
  }

  if (axes.size() != std::set<std::int32_t>{axes.begin(), axes.end()}.size()) {
    throw std::invalid_argument{"Duplicate axes found"};
  }

  for (auto&& ax_i : axes) {
    if (ax_i < 0 || ax_i >= static_cast<std::int32_t>(dim())) {
      throw std::invalid_argument{fmt::format("Invalid axis {} for a {}-D store", ax_i, dim())};
    }
  }

  auto new_extents = extents().map(axes);
  auto transform   = stack(transform_, std::make_unique<Transpose>(std::move(axes)));
  return make_internal_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::delinearize(std::int32_t dim,
                                                          std::vector<std::uint64_t> sizes)
{
  if (dim < 0 || dim >= static_cast<std::int32_t>(this->dim())) {
    throw std::invalid_argument{
      fmt::format("Invalid delinearization on dimension {} for a {}-D store", dim, this->dim())};
  }

  auto&& old_extents = extents();

  auto delinearizable = [](auto&& to_match, const auto& new_dim_extents) {
    auto begin = new_dim_extents.begin();
    auto end   = new_dim_extents.end();
    return std::reduce(begin, end, size_t{1}, std::multiplies<>{}) == to_match &&
           // overflow check
           std::all_of(begin, end, [&to_match](auto size) { return size <= to_match; });
  };

  if (!delinearizable(old_extents[dim], sizes)) {
    throw std::invalid_argument{
      fmt::format("Dimension of size {} cannot be delinearized into {}", old_extents[dim], sizes)};
  }

  auto new_extents = tuple<std::uint64_t>{};

  new_extents.reserve(old_extents.size() + sizes.size() - 1);
  for (int i = 0; i < dim; i++) {
    new_extents.append_inplace(old_extents[i]);
  }
  for (auto&& size : sizes) {
    new_extents.append_inplace(size);
  }
  for (auto i = static_cast<std::uint32_t>(dim + 1); i < old_extents.size(); i++) {
    new_extents.append_inplace(old_extents[i]);
  }

  auto transform = stack(transform_, std::make_unique<Delinearize>(dim, std::move(sizes)));
  return make_internal_shared<LogicalStore>(std::move(new_extents), storage_, std::move(transform));
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::partition_by_tiling_(
  const InternalSharedPtr<LogicalStore>& self, tuple<std::uint64_t> tile_shape)
{
  LEGATE_ASSERT(self.get() == this);
  if (tile_shape.size() != dim()) {
    throw std::invalid_argument{
      fmt::format("Incompatible tile shape: expected a {}-tuple, got a {}-tuple",
                  extents().size(),
                  tile_shape.size())};
  }
  if (tile_shape.volume() == 0) {
    throw std::invalid_argument{"Tile shape must have a volume greater than 0"};
  }
  auto color_shape = apply([](auto c, auto t) { return (c + t - 1) / t; }, extents(), tile_shape);
  auto partition   = create_tiling(std::move(tile_shape), std::move(color_shape));
  return create_partition_(self, std::move(partition), true);
}

InternalSharedPtr<PhysicalStore> LogicalStore::get_physical_store()
{
  if (unbound()) {
    throw std::invalid_argument{"Unbound store cannot be inlined mapped"};
  }
  if (mapped_) {
    return mapped_;
  }
  auto* storage = get_storage();
  if (storage->kind() == Storage::Kind::FUTURE ||
      (storage->kind() == Storage::Kind::FUTURE_MAP && storage->replicated())) {
    // TODO(wonchanl): future wrappers from inline mappings are read-only for now
    auto domain = to_domain(storage->shape()->extents());
    auto future =
      FutureWrapper{true, type()->size(), storage->scalar_offset(), domain, storage->get_future()};
    // Physical stores for future-backed stores shouldn't be cached, as they are not automatically
    // remapped to reflect changes by the runtime.
    return make_internal_shared<PhysicalStore>(dim(), type(), -1, std::move(future), transform_);
  }

  LEGATE_ASSERT(storage->kind() == Storage::Kind::REGION_FIELD);
  auto region_field = storage->map();
  mapped_ =
    make_internal_shared<PhysicalStore>(dim(), type(), -1, std::move(region_field), transform_);
  return mapped_;
}

// Just because all the member functions used are const, doesn't mean this function
// is. Detaching a decidedly non-const operation.
// NOLINTNEXTLINE(readability-make-member-function-const)
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
    get_storage()->allow_out_of_order_destruction();
  }
}

Restrictions LogicalStore::compute_restrictions(bool is_output) const
{
  return transform_->convert(get_storage()->compute_restrictions(), is_output);
}

Legion::ProjectionID LogicalStore::compute_projection(
  const Domain& launch_domain,
  const tuple<std::uint64_t>& color_shape,
  const std::optional<SymbolicPoint>& projection) const
{
  // If this is a sequential launch, the projection functor id is always 0
  if (!launch_domain.is_valid()) {
    return 0;
  }

  const auto launch_ndim = static_cast<std::uint32_t>(launch_domain.dim);
  const auto runtime     = Runtime::get_runtime();

  // If there's a custom projection, we just query its functor id
  if (projection) {
    // TODO(wonchanl): we should check if the projection is valid for the launch domain
    // (i.e., projection->size() == launch_ndim)
    return runtime->get_affine_projection(launch_ndim, transform_->invert(*projection));
  }

  // We are about to generate a projection functor
  const auto ndim = dim();

  // Easy case where the store and launch domain have the same number of dimensions
  if (ndim == launch_ndim) {
    return transform_->identity() ? 0
                                  : runtime->get_affine_projection(
                                      ndim, transform_->invert(proj::create_symbolic_point(ndim)));
  }

  // If we're here, it means the launch domain has to be 1D due to mixed store dimensionalities
  LEGATE_ASSERT(launch_ndim == 1);

  // Check if the color shape has only one dimension of a non-unit extent, in which case
  // delinearization would simply be projections
  if (std::count_if(
        color_shape.begin(), color_shape.end(), [](const auto& ext) { return ext != 1; }) == 1) {
    SymbolicPoint embed_1d_to_nd;

    embed_1d_to_nd.reserve(color_shape.size());
    for (auto&& ext : color_shape) {
      embed_1d_to_nd.append_inplace(ext != 1 ? dimension(0) : constant(0));
    }

    return runtime->get_affine_projection(launch_ndim,
                                          transform_->invert(std::move(embed_1d_to_nd)));
  }

  // When the store wasn't transformed, we could simply return the top-level delinearizing functor
  return transform_->identity()
           ? runtime->get_delinearizing_projection(color_shape)
           : runtime->get_compound_projection(
               color_shape, transform_->invert(proj::create_symbolic_point(ndim)));
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
    storage_part = get_storage()->find_key_partition(machine, transform_->invert(restrictions));
  }

  std::unique_ptr<Partition> store_part{};
  if (nullptr == storage_part || (!transform_->identity() && !storage_part->is_convertible())) {
    auto&& exts       = extents();
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
    LEGATE_ASSERT(store_part);
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
         get_storage()->find_key_partition(machine, transform_->invert(restrictions)) != nullptr;
}

void LogicalStore::set_key_partition(const mapping::detail::Machine& machine,
                                     const Partition* partition)
{
  num_pieces_ = machine.count();
  key_partition_.reset(partition->clone().release());
  get_storage()->set_key_partition(machine, transform_->invert(partition));
}

void LogicalStore::reset_key_partition()
{
  // Need to flush scheduling window to make this effective
  Runtime::get_runtime()->flush_scheduling_window();
  key_partition_.reset();
  get_storage()->reset_key_partition();
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::create_partition_(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  LEGATE_ASSERT(self.get() == this);
  if (unbound()) {
    throw std::invalid_argument{"Unbound store cannot be manually partitioned"};
  }
  auto storage_partition =
    get_storage()->create_partition(transform_->invert(partition.get()), std::move(complete));
  return make_internal_shared<LogicalStorePartition>(
    std::move(partition), std::move(storage_partition), self);
}

void LogicalStore::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(has_scalar_storage());
  buffer.pack<bool>(unbound());
  buffer.pack<std::uint32_t>(dim());
  type()->pack(buffer);
  transform_->pack(buffer);
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg_(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  std::int64_t redop)
{
  LEGATE_ASSERT(self.get() == this);

  switch (get_storage()->kind()) {
    case Storage::Kind::FUTURE: {
      return future_to_launcher_arg_(get_storage()->get_future(), launch_domain, privilege, redop);
    }
    case Storage::Kind::FUTURE_MAP: {
      return future_map_to_launcher_arg_(launch_domain, privilege, redop);
    }
    case Storage::Kind::REGION_FIELD: {
      return region_field_to_launcher_arg_(
        self, variable, strategy, launch_domain, projection, privilege, redop);
    }
  }

  LEGATE_UNREACHABLE();
  return nullptr;
}

std::unique_ptr<Analyzable> LogicalStore::future_to_launcher_arg_(Legion::Future future,
                                                                  const Domain& launch_domain,
                                                                  Legion::PrivilegeMode privilege,
                                                                  std::int64_t redop)
{
  if (!launch_domain.is_valid() && LEGION_REDUCE == privilege) {
    privilege = LEGION_READ_WRITE;
  }

  if (!future.exists()) {
    if (privilege != LEGION_WRITE_ONLY) {
      throw std::invalid_argument{
        "Read access or reduction to an uninitialized scalar store is prohibited"};
    }
    return std::make_unique<WriteOnlyScalarStoreArg>(this, redop);
  }

  // Scalar reductions don't need to pass the future or future map holding the current value to the
  // task, as the physical stores will be initialized with the reduction identity. They are later
  // passed to a future map reduction as an initial value in the task launch postamble.
  if (privilege == LEGION_REDUCE) {
    return std::make_unique<WriteOnlyScalarStoreArg>(this, redop);
  }

  // TODO(wonchanl): technically, we can create a WriteOnlyScalarStoreArg when privilege is
  // LEGION_WRITE_ONLY. Unfortunately, we don't currently track scalar stores passed as both inputs
  // and outputs, which are currently mapped to separate physical stores in the task. So, the
  // privilege of this store alone doesn't tell us whether it's truly a write-only store or this is
  // also passed as an input store. For the time being, we just pass the future when it exists even
  // when the store is not actually read by the task.
  return std::make_unique<ScalarStoreArg>(
    this, std::move(future), get_storage()->scalar_offset(), privilege == LEGION_READ_ONLY, redop);
}

std::unique_ptr<Analyzable> LogicalStore::future_map_to_launcher_arg_(
  const Domain& launch_domain, Legion::PrivilegeMode privilege, std::int64_t redop)
{
  if (unbound()) {
    return std::make_unique<WriteOnlyScalarStoreArg>(this, -1 /*redop*/);
  }
  LEGATE_ASSERT(get_storage()->replicated());

  auto future_or_future_map = get_storage()->get_future_or_future_map(launch_domain);

  return std::visit(
    [&](auto&& val) -> std::unique_ptr<Analyzable> {
      using T = std::decay_t<decltype(val)>;
      if constexpr (std::is_same_v<T, Legion::Future>) {
        return future_to_launcher_arg_(std::forward<T>(val), launch_domain, privilege, redop);
      }
      if constexpr (std::is_same_v<T, Legion::FutureMap>) {
        // Scalar reductions don't need to pass the future or future map holding the current value
        // to the task, as the physical stores will be initialized with the reduction identity. They
        // are later passed to a future map reduction as an initial value in the task launch
        // postamble.
        if (privilege == LEGION_REDUCE) {
          return std::make_unique<WriteOnlyScalarStoreArg>(this, redop);
        }
        // TODO(wonchanl): technically, we can create a WriteOnlyScalarStoreArg when privilege is
        // LEGION_WRITE_ONLY. Unfortunately, we don't currently track scalar stores passed as both
        // inputs and outputs, which are currently mapped to separate physical stores in the task.
        // So, the privilege of this store alone doesn't tell us whether it's truly a write-only
        // store or this is also passed as an input store. For the time being, we just pass the
        // future when it exists even when the store is not actually read by the task.
        return std::make_unique<ReplicatedScalarStoreArg>(this,
                                                          std::forward<T>(val),
                                                          get_storage()->scalar_offset(),
                                                          privilege == LEGION_READ_ONLY);
      }
      LEGATE_UNREACHABLE();
      return nullptr;
    },
    future_or_future_map);
}

std::unique_ptr<Analyzable> LogicalStore::region_field_to_launcher_arg_(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  std::int64_t redop)
{
  if (unbound()) {
    auto&& [field_space, field_id] = strategy.find_field_for_unbound_store(variable);
    return std::make_unique<OutputRegionArg>(this, field_space, field_id);
  }

  auto&& partition     = strategy[variable];
  auto store_partition = create_partition_(self, partition);
  auto store_proj      = store_partition->create_store_projection(launch_domain, projection);
  store_proj->is_key   = strategy.is_key_partition(variable);
  store_proj->redop    = static_cast<Legion::ReductionOpID>(redop);

  if (privilege == LEGION_REDUCE && store_partition->is_disjoint_for(launch_domain)) {
    privilege = LEGION_READ_WRITE;
  }
  if ((privilege == LEGION_WRITE_ONLY || privilege == LEGION_READ_WRITE) &&
      partition->has_launch_domain()) {
    set_key_partition(variable->operation()->machine(), partition.get());
  }

  return std::make_unique<RegionFieldArg>(this, privilege, std::move(store_proj));
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg_for_fixup_(
  const InternalSharedPtr<LogicalStore>& self,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege)
{
  LEGATE_ASSERT(self.get() == this);
  LEGATE_ASSERT(self->key_partition_ != nullptr);
  LEGATE_ASSERT(get_storage()->kind() == Storage::Kind::REGION_FIELD);
  auto store_partition = create_partition_(self, self->key_partition_);
  auto store_proj      = store_partition->create_store_projection(launch_domain);
  return std::make_unique<RegionFieldArg>(this, privilege, std::move(store_proj));
}

std::string LogicalStore::to_string() const
{
  auto result = fmt::format("Store({}) {{shape: ", store_id_);

  if (unbound()) {
    fmt::format_to(std::back_inserter(result), "(unbound)");
  } else {
    fmt::format_to(std::back_inserter(result), "{}", extents());
  }
  if (!transform_->identity()) {
    fmt::format_to(std::back_inserter(result), ", transform: {}", fmt::streamed(*transform_));
  }
  fmt::format_to(std::back_inserter(result), ", storage: {}}}", get_storage()->id());
  return result;
}

bool LogicalStore::equal_storage(const LogicalStore& other) const
{
  if (this == &other) {
    // fast path
    return true;
  }

  const auto kind = get_storage()->kind();

  if (kind != other.get_storage()->kind()) {
    return false;
  }

  switch (kind) {
    case Storage::Kind::REGION_FIELD: {
      auto&& rf       = get_region_field();
      auto&& other_rf = other.get_region_field();
      return rf->field_id() == other_rf->field_id() && rf->region() == other_rf->region();
    }
    case Storage::Kind::FUTURE: return get_future() == other.get_future();
    case Storage::Kind::FUTURE_MAP: return get_future_map() == other.get_future_map();
  }
  // Because sometimes, GCC is really stupid:
  //
  // error: control reaches end of non-void function [-Werror=return-type]
  LEGATE_UNREACHABLE();
  return false;
}

////////////////////////////////////////////////////
// legate::detail::LogicalStorePartition
////////////////////////////////////////////////////

InternalSharedPtr<LogicalStore> LogicalStorePartition::get_child_store(
  const tuple<std::uint64_t>& color) const
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw std::runtime_error{"Child stores can be retrieved only from tile partitions"};
  }
  const auto* tiling = static_cast<const Tiling*>(partition_.get());

  if (!tiling->has_color(color)) {
    throw std::out_of_range{
      fmt::format("Color {} is invalid for partition of color shape {}", color, color_shape())};
  }

  auto transform = store_->transform();
  // TODO(jfaibussowit)
  // Can move color here
  auto inverted_color = transform->invert_color(color);
  auto child_storage  = storage_partition_->get_child_storage(inverted_color);

  auto child_extents = tiling->get_child_extents(store_->extents(), inverted_color);
  auto child_offsets = tiling->get_child_offsets(inverted_color);

  for (auto&& [dim, coff] : legate::detail::enumerate(child_offsets)) {
    if (coff != 0) {
      transform = make_internal_shared<TransformStack>(std::make_unique<Shift>(dim, -coff),
                                                       std::move(transform));
    }
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
  auto proj_id = store_->compute_projection(launch_domain, partition_->color_shape(), projection);
  return std::make_unique<StoreProjection>(std::move(legion_partition), proj_id);
}

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return storage_partition_->is_disjoint_for(launch_domain);
}

const tuple<std::uint64_t>& LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

}  // namespace legate::detail
