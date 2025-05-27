/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store.h>

#include <legate_defines.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/transform.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/task_return_layout.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace legate::detail {

////////////////////////////////////////////////////
// legate::detail::Storage
////////////////////////////////////////////////////

Storage::Storage(InternalSharedPtr<Shape> shape,
                 std::uint32_t field_size,
                 bool optimize_scalar,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    unbound_{shape->unbound()},
    shape_{std::move(shape)},
    kind_{[&] {
      if (optimize_scalar) {
        if (unbound()) {
          return Kind::FUTURE_MAP;
        }
        // We should not blindly check the shape volume as it would flush the scheduling window
        if (this->shape()->ready() && (this->shape()->volume() == 1)) {
          return Kind::FUTURE;
        }
      }
      return Kind::REGION_FIELD;
    }()},
    provenance_{std::move(provenance)},
    offsets_{legate::full(dim(), std::int64_t{0})}
{
  if (kind_ == Kind::REGION_FIELD && !unbound()) {
    auto&& runtime = Runtime::get_runtime();

    region_field_ = runtime.create_region_field(this->shape(), field_size);
    runtime.attach_alloc_info(get_region_field(), this->provenance());
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(InternalSharedPtr<Shape> shape, Legion::Future future, std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    shape_{std::move(shape)},
    kind_{Kind::FUTURE},
    provenance_{std::move(provenance)},
    offsets_{legate::full(dim(), std::int64_t{0})},
    future_{std::move(future)}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(tuple<std::uint64_t> extents,
                 InternalSharedPtr<StoragePartition> parent,
                 tuple<std::uint64_t> color,
                 tuple<std::int64_t> offsets)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    color_{std::move(color)},
    offsets_{std::move(offsets)},
    region_field_{(*parent_)->get_child_data(color_)}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

// Leak is intentional
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
Storage::~Storage()
{
  if (!has_started()) {
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

const tuple<std::int64_t>& Storage::offsets() const
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
    auto lext = static_cast<std::int64_t>(lexts[idx]);
    auto rext = static_cast<std::int64_t>(rexts[idx]);
    auto loff = lhs->offsets_[idx];
    auto roff = rhs->offsets_[idx];

    if (loff <= roff ? roff < loff + lext : loff < roff + rext) {
      continue;
    }
    return false;
  }
  return true;
}

bool Storage::is_mapped() const
{
  // TODO(wonchanl): future- and future map-backed storages are considered unmapped until we
  // implement the full state machine for them (they are currently read only)
  return !unbound() && kind() == Kind::REGION_FIELD && get_region_field()->is_mapped();
}

InternalSharedPtr<Storage> Storage::slice(const InternalSharedPtr<Storage>& self,
                                          tuple<std::uint64_t> tile_shape,
                                          tuple<std::int64_t> offsets)
{
  LEGATE_ASSERT(self.get() == this);

  if (Kind::FUTURE == kind()) {
    return self;
  }

  const auto root   = get_root(self);
  const auto& shape = root->extents();
  const auto can_tile_completely =
    (shape % tile_shape).sum() == 0 && (offsets % tile_shape).sum() == 0 &&
    Runtime::get_runtime().partition_manager().use_complete_tiling(shape, tile_shape);

  tuple<std::uint64_t> color_shape, color;
  if (can_tile_completely) {
    color_shape = shape / tile_shape;
    color       = offsets / tile_shape;
    std::fill(offsets.begin(), offsets.end(), 0);
  } else {
    color_shape = legate::full<std::uint64_t>(shape.size(), 1);
    color       = legate::full<std::uint64_t>(shape.size(), 0);
  }

  auto tiling = create_tiling(std::move(tile_shape), std::move(color_shape), std::move(offsets));
  auto storage_partition = root->create_partition(root, std::move(tiling), can_tile_completely);
  return storage_partition->get_child_storage(storage_partition, std::move(color));
}

const Storage* Storage::get_root() const
{
  return parent_.has_value() ? (*parent_)->get_root() : this;
}

Storage* Storage::get_root() { return parent_.has_value() ? (*parent_)->get_root() : this; }

InternalSharedPtr<const Storage> Storage::get_root(
  const InternalSharedPtr<const Storage>& self) const
{
  LEGATE_ASSERT(self.get() == this);
  return parent_.has_value() ? (*parent_)->get_root(*parent_) : self;
}

InternalSharedPtr<Storage> Storage::get_root(const InternalSharedPtr<Storage>& self)
{
  LEGATE_ASSERT(self.get() == this);
  return parent_.has_value() ? (*parent_)->get_root(*parent_) : self;
}

const InternalSharedPtr<LogicalRegionField>& Storage::get_region_field() const noexcept
{
  LEGATE_CHECK(kind_ == Kind::REGION_FIELD);
  LEGATE_CHECK(region_field_.has_value());
  return *region_field_;
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
  LEGATE_CHECK(future_map_.has_value());
  return *future_map_;
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
  return Runtime::get_runtime().reshape_future_map(future_map, launch_domain);
}

void Storage::set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field)
{
  LEGATE_CHECK(unbound_ && !region_field_.has_value());
  LEGATE_CHECK(!parent_.has_value());

  unbound_      = false;
  region_field_ = std::move(region_field);
  if (destroyed_out_of_order_) {
    (*region_field_)->allow_out_of_order_destruction();
  }
  Runtime::get_runtime().attach_alloc_info(*region_field_, provenance());
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

RegionField Storage::map(legate::mapping::StoreTarget target)
{
  LEGATE_ASSERT(Kind::REGION_FIELD == kind_);
  auto&& region_field = get_region_field();
  auto mapped         = region_field->map(target);
  // Set the right subregion so the physical store can see the right domain
  mapped.set_logical_region(region_field->region());
  return mapped;
}

void Storage::unmap()
{
  LEGATE_ASSERT(Kind::REGION_FIELD == kind_);
  get_region_field()->unmap();
}

void Storage::allow_out_of_order_destruction()
{
  // Technically speaking this property only needs to be tracked on (root) LogicalRegionFields, but
  // a Storage may not have instantiated its region_field_ yet, so we note this also on the (root)
  // Storage, in case we need to propagate later. We only need to note this on the root Storage,
  // because any call that sets region_field_ (get_region_field(), set_region_field()) will end up
  // touching the root Storage.
  if (parent_.has_value()) {
    get_root()->allow_out_of_order_destruction();
  } else if (!destroyed_out_of_order_) {
    destroyed_out_of_order_ = true;
    if (region_field_.has_value()) {
      (*region_field_)->allow_out_of_order_destruction();
    }
  }
}

void Storage::free_early() noexcept
{
  if (kind_ != Kind::REGION_FIELD || unbound()) {
    return;
  }
  get_region_field()->release_region_field();
}

Restrictions Storage::compute_restrictions() const
{
  return legate::full<Restriction>(dim(), Restriction::ALLOW);
}

std::optional<InternalSharedPtr<Partition>> Storage::find_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((num_pieces_ == new_num_pieces) && key_partition_.has_value() &&
      (*key_partition_)->satisfies_restrictions(restrictions)) {
    return key_partition_;
  }
  if (parent_.has_value()) {
    return (*parent_)->find_key_partition(machine, parallel_policy, restrictions);
  }
  return std::nullopt;
}

void Storage::set_key_partition(const mapping::detail::Machine& machine,
                                InternalSharedPtr<Partition> key_partition)
{
  num_pieces_    = machine.count();
  key_partition_ = std::move(key_partition);
}

void Storage::reset_key_partition() noexcept { key_partition_.reset(); }

InternalSharedPtr<StoragePartition> Storage::create_partition(
  const InternalSharedPtr<Storage>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  LEGATE_ASSERT(self.get() == this);
  if (!complete.has_value()) {
    complete = partition->is_complete_for(*this);
  }
  return make_internal_shared<StoragePartition>(self, std::move(partition), *complete);
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

  auto result = fmt::format("Storage({}) {{kind: {}, level: {}", id(), kind_str, level());

  if (kind() == Kind::REGION_FIELD) {
    if (unbound()) {
      result += ", region: unbound";
    } else {
      fmt::format_to(std::back_inserter(result),
                     ", region: {}, field: {}",
                     *get_region_field(),
                     get_region_field()->field_id());
    }
  }
  result += '}';
  return result;
}

////////////////////////////////////////////////////
// legate::detail::StoragePartition
////////////////////////////////////////////////////

const Storage* StoragePartition::get_root() const { return parent_->get_root(); }

Storage* StoragePartition::get_root() { return parent_->get_root(); }

InternalSharedPtr<const Storage> StoragePartition::get_root(
  const InternalSharedPtr<const StoragePartition>&) const
{
  return parent_->get_root(parent_);
}

InternalSharedPtr<Storage> StoragePartition::get_root(const InternalSharedPtr<StoragePartition>&)
{
  return parent_->get_root(parent_);
}

InternalSharedPtr<Storage> StoragePartition::get_child_storage(
  const InternalSharedPtr<StoragePartition>& self, tuple<std::uint64_t> color)
{
  LEGATE_ASSERT(self.get() == this);

  if (partition_->kind() != Partition::Kind::TILING) {
    throw TracedException<std::runtime_error>{"Sub-storage is implemented only for tiling"};
  }

  auto tiling        = static_cast<Tiling*>(partition_.get());
  auto child_extents = tiling->get_child_extents(parent_->extents(), color);
  auto child_offsets = tiling->get_child_offsets(color);
  return make_internal_shared<Storage>(
    std::move(child_extents), self, std::move(color), std::move(child_offsets));
}

InternalSharedPtr<LogicalRegionField> StoragePartition::get_child_data(
  const tuple<std::uint64_t>& color)
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw TracedException<std::runtime_error>{"Sub-storage is implemented only for tiling"};
  }

  auto tiling = static_cast<Tiling*>(partition_.get());
  return parent_->get_region_field()->get_child(tiling, color, complete_);
}

std::optional<InternalSharedPtr<Partition>> StoragePartition::find_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions) const
{
  return parent_->find_key_partition(machine, parallel_policy, restrictions);
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

void assert_fixed_storage_size(const InternalSharedPtr<Type>& type)
{
  if (type->variable_size()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Store cannot be created with variable size type {}", *type)};
  }
}

[[nodiscard]] InternalSharedPtr<TransformStack> stack(
  const InternalSharedPtr<TransformStack>& parent, std::unique_ptr<StoreTransform>&& transform)
{
  return make_internal_shared<TransformStack>(std::move(transform), parent);
}

}  // namespace

LogicalStore::LogicalStore(InternalSharedPtr<Storage> storage, InternalSharedPtr<Type> type)
  : store_id_{Runtime::get_runtime().get_unique_store_id()},
    type_{std::move(type)},
    shape_{storage->shape()},
    storage_{std::move(storage)},
    transform_{make_internal_shared<TransformStack>()}
{
  assert_fixed_storage_size(this->type());
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

LogicalStore::LogicalStore(tuple<std::uint64_t> extents,
                           InternalSharedPtr<Storage> storage,
                           InternalSharedPtr<Type> type,
                           InternalSharedPtr<TransformStack> transform)
  : store_id_{Runtime::get_runtime().get_unique_store_id()},
    type_{std::move(type)},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    storage_{std::move(storage)},
    transform_{std::move(transform)}
{
  assert_fixed_storage_size(this->type());
  LEGATE_ASSERT(transform_ != nullptr);
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::set_region_field(InternalSharedPtr<LogicalRegionField> region_field)
{
  LEGATE_ASSERT(!has_scalar_storage());
  // The shape object of this store must be an alias to that of the storage
  LEGATE_ASSERT(get_storage()->shape() == shape());
  // this call updates the shape for both the storage and the store
  get_storage()->set_region_field(std::move(region_field));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::set_future(Legion::Future future, std::size_t scalar_offset)
{
  LEGATE_ASSERT(has_scalar_storage());
  get_storage()->set_future(std::move(future), scalar_offset);
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset)
{
  LEGATE_ASSERT(has_scalar_storage());
  get_storage()->set_future_map(std::move(future_map), scalar_offset);
}

InternalSharedPtr<LogicalStore> LogicalStore::reinterpret_as(InternalSharedPtr<Type> type) const
{
  if (type->size() != this->type()->size()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Cannot reinterpret a {}-backed store as {}, size of the types must be equal",
                  *type,
                  *(this->type()))};
  }
  if (type->alignment() != this->type()->alignment()) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Cannot reinterpret a {}-backed store as {}, alignment of the types must be equal",
      *type,
      *(this->type()))};
  }
  return make_internal_shared<LogicalStore>(get_storage(), std::move(type));
}

InternalSharedPtr<LogicalStore> LogicalStore::promote(std::int32_t extra_dim, std::size_t dim_size)
{
  if (extra_dim < 0 || extra_dim > static_cast<std::int32_t>(dim())) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid promotion on dimension {} for a {}-D store", extra_dim, dim())};
  }

  auto new_extents = extents().insert(extra_dim, dim_size);
  auto transform   = stack(transform_, std::make_unique<Promote>(extra_dim, dim_size));
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::project(std::int32_t d, std::int64_t index)
{
  auto&& old_extents = extents();

  if (d < 0 || d >= static_cast<std::int32_t>(dim())) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid projection on dimension {} for a {}-D store", d, dim())};
  }

  if (index < 0 || index >= static_cast<std::int64_t>(old_extents[d])) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Projection index {} is out of bounds [0, {})", index, old_extents[d])};
  }

  auto new_extents = old_extents.remove(d);
  auto transform   = stack(transform_, std::make_unique<Project>(d, index));
  auto substorage =
    volume() == 0
      ? storage_
      : slice_storage(storage_,
                      transform->invert_extents(new_extents),
                      transform->invert_point(legate::full<std::int64_t>(new_extents.size(), 0)));
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), std::move(substorage), type(), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::slice_(const InternalSharedPtr<LogicalStore>& self,
                                                     std::int32_t dim,
                                                     Slice slice)
{
  LEGATE_ASSERT(self.get() == this);
  if (dim < 0 || dim >= static_cast<std::int32_t>(this->dim())) {
    throw TracedException<std::invalid_argument>{
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

    return std::make_pair<std::size_t, std::size_t>(std::max(std::int64_t{0}, start),
                                                    std::max(std::int64_t{0}, stop));
  };

  auto exts          = extents();
  auto [start, stop] = sanitize_slice(slice, static_cast<std::int64_t>(exts[dim]));

  if (start < stop && (start >= exts[dim] || stop > exts[dim])) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Out-of-bounds slicing [{}, {}] on dimension {} for a store of shape {}",
                  start,
                  stop,
                  dim,
                  extents())};
  }

  exts[dim] = (start < stop) ? (stop - start) : 0;

  if (exts[dim] == extents()[dim]) {
    return self;
  }

  if (0 == exts[dim]) {
    return Runtime::get_runtime().create_store(
      make_internal_shared<Shape>(std::move(exts)), type(), false /*optimize_scalar*/);
  }

  auto transform =
    (start == 0) ? transform_ : stack(transform_, std::make_unique<Shift>(dim, -start));
  auto substorage =
    volume() == 0
      ? storage_
      : slice_storage(storage_,
                      transform->invert_extents(exts),
                      transform->invert_point(legate::full<std::int64_t>(exts.size(), 0)));
  return make_internal_shared<LogicalStore>(
    std::move(exts), std::move(substorage), type(), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::transpose(std::vector<std::int32_t> axes)
{
  if (axes.size() != dim()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Dimension Mismatch: expected {} axes, but got {}", dim(), axes.size())};
  }

  if (axes.size() != std::unordered_set<std::int32_t>{axes.begin(), axes.end()}.size()) {
    throw TracedException<std::invalid_argument>{"Duplicate axes found"};
  }

  for (auto&& ax_i : axes) {
    if (ax_i < 0 || ax_i >= static_cast<std::int32_t>(dim())) {
      throw TracedException<std::invalid_argument>{
        fmt::format("Invalid axis {} for a {}-D store", ax_i, dim())};
    }
  }

  auto new_extents = extents().map(axes);
  auto transform   = stack(transform_, std::make_unique<Transpose>(std::move(axes)));
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::delinearize(std::int32_t dim,
                                                          std::vector<std::uint64_t> sizes)
{
  if (dim < 0 || dim >= static_cast<std::int32_t>(this->dim())) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid delinearization on dimension {} for a {}-D store", dim, this->dim())};
  }

  auto&& old_extents = extents();

  auto delinearizable = [](auto&& to_match, const auto& new_dim_extents) {
    auto begin = new_dim_extents.begin();
    auto end   = new_dim_extents.end();
    return std::reduce(begin, end, std::size_t{1}, std::multiplies<>{}) == to_match &&
           // overflow check
           std::all_of(begin, end, [&to_match](auto size) { return size <= to_match; });
  };

  if (!delinearizable(old_extents[dim], sizes)) {
    throw TracedException<std::invalid_argument>{
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
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform));
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::partition_by_tiling_(
  const InternalSharedPtr<LogicalStore>& self, tuple<std::uint64_t> tile_shape)
{
  LEGATE_ASSERT(self.get() == this);
  if (tile_shape.size() != dim()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Incompatible tile shape: expected a {}-tuple, got a {}-tuple",
                  extents().size(),
                  tile_shape.size())};
  }
  if (tile_shape.volume() == 0) {
    throw TracedException<std::invalid_argument>{"Tile shape must have a volume greater than 0"};
  }
  auto color_shape = apply([](auto c, auto t) { return (c + t - 1) / t; }, extents(), tile_shape);
  auto partition   = create_tiling(std::move(tile_shape), std::move(color_shape));
  return create_partition_(self, std::move(partition), true);
}

InternalSharedPtr<PhysicalStore> LogicalStore::get_physical_store(
  legate::mapping::StoreTarget target, bool ignore_future_mutability)
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound store cannot be inlined mapped"};
  }

  // If there's already a physical store for this logical store, just return the cached one.
  // Any operations using this logical store will immediately flush the scheduling window.
  if (mapped_) {
    if (mapped_->on_target(target)) {
      return mapped_;
    }
    get_storage()->unmap();
    mapped_.reset();
  }

  // Otherwise, a physical allocation of this store is escaping to the user land for the first time,
  // so we need to make sure all outstanding operations are launched
  Runtime::get_runtime().flush_scheduling_window();

  auto&& storage = get_storage();
  if (storage->kind() == Storage::Kind::FUTURE ||
      (storage->kind() == Storage::Kind::FUTURE_MAP && storage->replicated())) {
    // TODO(wonchanl): future wrappers from inline mappings are read-only for now
    auto domain = [&]() -> Domain {
      auto&& extents = storage->extents();

      // Workaround needed for the inline-task fast-path. When empty domains are serialized,
      // they come out as (0-ed out), 0D Domains when deserialized. However, to_domain() will
      // produce a {0, 0} domain, i.e. still 0-ed out, but this this time it is 1D. So to match
      // the serialization case, we return a empty 0D domain here.
      if (extents.empty()) {
        return {};
      }
      return to_domain(extents);
    }();

    // If we ignore the mutability of the future, then this future is writable. Needed by the
    // inline-task fast-path
    auto future = FutureWrapper{/* read_only */ !ignore_future_mutability,
                                type()->size(),
                                type()->alignment(),
                                storage->scalar_offset(),
                                domain,
                                storage->get_future()};
    // Physical stores for future-backed stores shouldn't be cached, as they are not automatically
    // remapped to reflect changes by the runtime.
    return make_internal_shared<PhysicalStore>(
      dim(), type(), GlobalRedopID{-1}, std::move(future), transform_);
  }

  LEGATE_ASSERT(storage->kind() == Storage::Kind::REGION_FIELD);
  auto region_field = storage->map(target);
  mapped_           = make_internal_shared<PhysicalStore>(
    dim(), type(), GlobalRedopID{-1}, std::move(region_field), transform_);
  return mapped_;
}

// Just because all the member functions used are const, doesn't mean this function
// is. Detaching a decidedly non-const operation.
// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::detach()
{
  if (transformed()) {
    throw TracedException<std::invalid_argument>{"Manual detach must be called on the root store"};
  }
  if (has_scalar_storage() || unbound()) {
    throw TracedException<std::invalid_argument>{
      "Only stores created with share=true can be manually detached"};
  }
  get_region_field()->detach();
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::allow_out_of_order_destruction()
{
  if (Runtime::get_runtime().consensus_match_required()) {
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
  auto&& runtime         = Runtime::get_runtime();

  // If there's a custom projection, we just query its functor id
  if (projection) {
    // TODO(wonchanl): we should check if the projection is valid for the launch domain
    // (i.e., projection->size() == launch_ndim)
    return runtime.get_affine_projection(launch_ndim, transform_->invert(*projection));
  }

  // We are about to generate a projection functor
  const auto ndim = dim();

  // Easy case where the store and launch domain have the same number of dimensions
  if (ndim == launch_ndim) {
    return transform_->identity() ? 0
                                  : runtime.get_affine_projection(
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

    return runtime.get_affine_projection(launch_ndim,
                                         transform_->invert(std::move(embed_1d_to_nd)));
  }

  // When the store wasn't transformed, we could simply return the top-level delinearizing functor
  return transform_->identity()
           ? runtime.get_delinearizing_projection(color_shape)
           : runtime.get_compound_projection(color_shape,
                                             transform_->invert(proj::create_symbolic_point(ndim)));
}

InternalSharedPtr<Partition> LogicalStore::find_or_create_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions)
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((num_pieces_ == new_num_pieces) && key_partition_.has_value() &&
      (*key_partition_)->satisfies_restrictions(restrictions)) {
    return *key_partition_;
  }

  if (has_scalar_storage() || dim() == 0 || volume() == 0) {
    return create_no_partition();
  }

  std::optional<InternalSharedPtr<Partition>> storage_part{};

  if (transform_->is_convertible()) {
    storage_part =
      get_storage()->find_key_partition(machine, parallel_policy, transform_->invert(restrictions));
  }

  if (!storage_part.has_value() ||
      (!transform_->identity() && !(*storage_part)->is_convertible())) {
    auto&& exts       = extents();
    auto&& part_mgr   = Runtime::get_runtime().partition_manager();
    auto launch_shape = part_mgr.compute_launch_shape(machine, parallel_policy, restrictions, exts);

    if (launch_shape.empty()) {
      return create_no_partition();
    }
    auto tile_shape = part_mgr.compute_tile_shape(exts, launch_shape);

    return create_tiling(std::move(tile_shape), std::move(launch_shape));
  }
  return (*storage_part)->convert(*storage_part, transform());
}

bool LogicalStore::has_key_partition(const mapping::detail::Machine& machine,
                                     const ParallelPolicy& parallel_policy,
                                     const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((new_num_pieces == num_pieces_) && key_partition_.has_value() &&
      (*key_partition_)->satisfies_restrictions(restrictions)) {
    return true;
  }
  return transform_->is_convertible() &&
         get_storage()
           ->find_key_partition(machine, parallel_policy, transform_->invert(restrictions))
           .has_value();
}

void LogicalStore::set_key_partition(const mapping::detail::Machine& machine,
                                     const ParallelPolicy& parallel_policy,
                                     InternalSharedPtr<Partition> partition)
{
  num_pieces_ = machine.count() * parallel_policy.overdecompose_factor();
  get_storage()->set_key_partition(machine, partition->invert(partition, transform()));
  key_partition_ = std::move(partition);
}

void LogicalStore::reset_key_partition()
{
  // Need to flush scheduling window to make this effective
  Runtime::get_runtime().flush_scheduling_window();
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
    throw TracedException<std::invalid_argument>{"Unbound store cannot be manually partitioned"};
  }
  auto storage_partition = create_storage_partition(
    get_storage(), partition->invert(partition, transform()), std::move(complete));
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

void LogicalStore::calculate_pack_size(TaskReturnLayoutForUnpack* layout) const
{
  if (has_scalar_storage()) {
    std::ignore = layout->next(type()->size(), type()->alignment());
  } else if (unbound()) {
    // The number of elements for each unbound store is stored in a buffer of type
    // std::size_t
    std::ignore = layout->next(sizeof(std::size_t), alignof(std::size_t));
  }
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg_(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  GlobalRedopID redop)
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
                                                                  GlobalRedopID redop)
{
  if (!launch_domain.is_valid() && LEGION_REDUCE == privilege) {
    privilege = LEGION_READ_WRITE;
  }

  if (!future.exists()) {
    if (privilege != LEGION_WRITE_ONLY) {
      throw TracedException<std::invalid_argument>{
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
  const Domain& launch_domain, Legion::PrivilegeMode privilege, GlobalRedopID redop)
{
  if (unbound()) {
    return std::make_unique<WriteOnlyScalarStoreArg>(this, GlobalRedopID{-1} /*redop*/);
  }
  LEGATE_ASSERT(get_storage()->replicated());

  auto future_or_future_map = get_storage()->get_future_or_future_map(launch_domain);

  return std::visit(
    Overload{
      [&](Legion::Future fut) {
        return future_to_launcher_arg_(std::move(fut), launch_domain, privilege, redop);
      },
      [&](Legion::FutureMap map) -> std::unique_ptr<Analyzable> {
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
        return std::make_unique<ReplicatedScalarStoreArg>(
          this, std::move(map), get_storage()->scalar_offset(), privilege == LEGION_READ_ONLY);
      }},
    future_or_future_map);
}

std::unique_ptr<Analyzable> LogicalStore::region_field_to_launcher_arg_(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  GlobalRedopID redop)
{
  if (unbound()) {
    auto&& [field_space, field_id] = strategy.find_field_for_unbound_store(variable);
    return std::make_unique<OutputRegionArg>(this, field_space, field_id);
  }

  auto&& partition     = strategy[variable];
  auto store_partition = create_partition_(self, partition);
  auto store_proj      = store_partition->create_store_projection(launch_domain, projection);
  store_proj->is_key   = strategy.is_key_partition(variable);
  store_proj->redop    = redop;

  if (privilege == LEGION_REDUCE && store_partition->is_disjoint_for(launch_domain)) {
    privilege = LEGION_READ_WRITE;
  }
  if ((privilege == LEGION_WRITE_ONLY || privilege == LEGION_READ_WRITE) &&
      partition->has_launch_domain()) {
    const auto* op = variable->operation();

    set_key_partition(op->machine(), op->parallel_policy(), partition);
  }

  return std::make_unique<RegionFieldArg>(this, privilege, std::move(store_proj));
}

std::unique_ptr<Analyzable> LogicalStore::to_launcher_arg_for_fixup_(
  const InternalSharedPtr<LogicalStore>& self,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege)
{
  LEGATE_ASSERT(self.get() == this);
  LEGATE_ASSERT(self->key_partition_.has_value());
  LEGATE_ASSERT(get_storage()->kind() == Storage::Kind::REGION_FIELD);
  auto store_partition =
    create_partition_(self,
                      *self->key_partition_  // NOLINT(bugprone-unchecked-optional-access)
    );
  auto store_proj = store_partition->create_store_projection(launch_domain);
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
    fmt::format_to(std::back_inserter(result), ", transform: {}", fmt::streamed(*transform()));
  }
  fmt::format_to(std::back_inserter(result), ", type: {}, storage: {}}}", *type(), *get_storage());
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
    case Storage::Kind::FUTURE: [[fallthrough]];
    case Storage::Kind::FUTURE_MAP: {
      // Future- and future map-backed stores are not sliced and thus cannot be aliased through
      // different storage objects, so equality on the storage is sufficient
      return get_storage() == other.get_storage();
    }
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
    throw TracedException<std::runtime_error>{
      "Child stores can be retrieved only from tile partitions"};
  }
  const auto* tiling = static_cast<const Tiling*>(partition_.get());

  if (!tiling->has_color(color)) {
    throw TracedException<std::out_of_range>{
      fmt::format("Color {} is invalid for partition of color shape {}", color, color_shape())};
  }

  auto transform = store_->transform();
  // TODO(jfaibussowit)
  // Can move color here
  auto inverted_color = transform->invert_color(color);
  auto child_storage  = storage_partition_->get_child_storage(storage_partition_, inverted_color);

  auto child_extents = tiling->get_child_extents(store_->extents(), inverted_color);
  auto child_offsets = tiling->get_child_offsets(inverted_color);

  for (auto&& [dim, coff] : legate::detail::enumerate(child_offsets)) {
    if (coff != 0) {
      transform = make_internal_shared<TransformStack>(std::make_unique<Shift>(dim, -coff),
                                                       std::move(transform));
    }
  }

  return make_internal_shared<LogicalStore>(
    std::move(child_extents), std::move(child_storage), store_->type(), std::move(transform));
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
