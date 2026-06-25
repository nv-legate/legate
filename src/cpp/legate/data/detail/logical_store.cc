/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store.h>

#include <legate_defines.h>

#include <legate/data/detail/data_movement_requests.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/transform/dim_broadcast.h>
#include <legate/data/detail/transform/project.h>
#include <legate/data/detail/transform/promote.h>
#include <legate/data/detail/transform/shift.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/data/detail/transform/transpose.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/detail/type_traits.h>

#include <legion/api/values.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace legate::detail {

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
    transform_{make_internal_shared<TransformStack>()},
    domain_{to_maybe_empty_domain(shape()->extents())}
{
  assert_fixed_storage_size(this->type());
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

LogicalStore::LogicalStore(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
                           InternalSharedPtr<Storage> storage,
                           InternalSharedPtr<Type> type,
                           InternalSharedPtr<TransformStack> transform,
                           Domain domain)
  : store_id_{Runtime::get_runtime().get_unique_store_id()},
    type_{std::move(type)},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    storage_{std::move(storage)},
    transform_{std::move(transform)},
    domain_{domain}
{
  assert_fixed_storage_size(this->type());
  LEGATE_ASSERT(transform_ != nullptr);
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
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

  auto new_extents = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{extents()};

  new_extents.insert(new_extents.begin() + extra_dim, dim_size);

  auto transform = stack(transform_, std::make_unique<Promote>(extra_dim, dim_size));

  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform), domain_);
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

  auto new_extents = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{old_extents};

  new_extents.erase(new_extents.begin() + d);

  auto transform = stack(transform_, std::make_unique<Project>(d, index));

  // New domain over Storage simply restricts to the single dimension in the projection.
  const auto projected_domain = [&] {
    DomainPoint lo = domain_.lo();
    DomainPoint hi = domain_.hi();

    lo[d] = index;
    hi[d] = index;
    return Domain{lo, hi};
  }();

  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform), projected_domain);
}

InternalSharedPtr<LogicalStore> LogicalStore::broadcast(std::int32_t bcast_dim,
                                                        std::size_t dim_size)
{
  if (bcast_dim < 0 || bcast_dim >= static_cast<std::int32_t>(dim())) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid broadcast on dimension {} for a {}-D store", bcast_dim, dim())};
  }
  if (extents()[bcast_dim] != 1) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid broadcast on dimension {}: expected size 1 but got {}",
                  bcast_dim,
                  extents()[bcast_dim])};
  }

  auto transform   = stack(transform_, std::make_unique<DimBroadcast>(bcast_dim, dim_size));
  auto new_extents = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{extents()};

  new_extents[bcast_dim] = dim_size;
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform), domain_);
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

  auto exts          = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{extents()};
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

  auto&& runtime = Runtime::get_runtime();

  if (0 == exts[dim]) {
    return runtime.create_store(make_internal_shared<Shape>(std::move(exts)), type());
  }

  auto transform =
    (start == 0) ? transform_ : stack(transform_, std::make_unique<Shift>(dim, -start));
  // Compute the new domain of the root storage that this LogicalStore covers.
  const auto sstart     = static_cast<std::int64_t>(start);
  const auto extent_dim = static_cast<std::int64_t>(exts[dim]);
  const auto subdomain  = [&] {
    DomainPoint lo = domain_.lo();
    DomainPoint hi = domain_.hi();

    lo[dim] += sstart;
    hi[dim] = std::min(lo[dim] + extent_dim - 1, hi[dim]);
    return Domain{lo, hi};
  }();

  return make_internal_shared<LogicalStore>(
    std::move(exts), storage_, type(), std::move(transform), subdomain);
}

InternalSharedPtr<LogicalStore> LogicalStore::transpose(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> axes)
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

  auto new_extents = array_map<SmallVector<std::uint64_t, LEGATE_MAX_DIM>>(extents(), axes);
  auto transform   = stack(transform_, std::make_unique<Transpose>(std::move(axes)));
  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform), domain_);
}

InternalSharedPtr<LogicalStore> LogicalStore::delinearize(
  std::int32_t, SmallVector<std::uint64_t, LEGATE_MAX_DIM>)
{
  // TODO(wonchanl): V2 needs to subsume reshaping
  LEGATE_ABORT("No support for delinearization");
  return nullptr;
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::partition_by_tiling_(
  const InternalSharedPtr<LogicalStore>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  std::optional<SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape)
{
  LEGATE_ASSERT(self.get() == this);
  if (tile_shape.size() != dim()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Incompatible tile shape: expected a {}-tuple, got a {}-tuple",
                  extents().size(),
                  tile_shape.size())};
  }
  if (array_volume(tile_shape) == 0) {
    throw TracedException<std::invalid_argument>{"Tile shape must have a volume greater than 0"};
  }
  if (color_shape.has_value()) {
    if (color_shape->size() != dim()) {
      throw TracedException<std::invalid_argument>{
        fmt::format("Incompatible color shape: expected a {}-tuple, got a {}-tuple",
                    extents().size(),
                    (*color_shape).size())};
    }
    if (array_volume(*color_shape) == 0) {
      throw TracedException<std::invalid_argument>{"Color shape must have a volume greater than 0"};
    }
  }

  auto partition = [&] {
    if (color_shape.has_value()) {
      return create_tiling(std::move(tile_shape), std::move(*color_shape));
    }
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tmp_color_shape;

    tmp_color_shape.reserve(extents().size());
    std::transform(extents().begin(),
                   extents().end(),
                   tile_shape.begin(),
                   std::back_inserter(tmp_color_shape),
                   [](std::uint64_t c, std::uint64_t t) { return (c + t - 1) / t; });
    return create_tiling(std::move(tile_shape), std::move(tmp_color_shape));
  }();

  return create_partition_(self, std::move(partition), /* complete */ true);
}

namespace {

DomainPoint compute_color(const SymbolicPoint& store_projection, const DomainPoint& task_index)
{
  if (store_projection.empty() || task_index.dim == 0) {
    return task_index;
  }

  DomainPoint result{};

  result.dim = static_cast<std::int32_t>(store_projection.size());
  for (auto&& [dim, expr] : enumerate(store_projection)) {
    result[dim] = expr.compute_coordinate(task_index);
  }

  return result;
}

}  // namespace

std::pair<InternalSharedPtr<PhysicalStore>, bool> LogicalStore::map(
  const InternalSharedPtr<Partition>& partition,
  const SymbolicPoint& store_projection,
  const Domain&,
  const DomainPoint& task_index,
  bool is_read,
  const legate::mapping::detail::Machine& machine,
  DataMovementRequests* requests)
{
  const auto color = compute_color(store_projection, task_index);

  auto inverted = partition->invert(partition, transform_);

  auto&& [allocation, subdomain, subsumed] = get_storage()->map(
    domain_, inverted, color, machine.preferred_store_target(), is_read, requests);

  return {
    make_internal_shared<PhysicalStore>(
      shape()->ndim(), type(), transform_, LEGION_READ_WRITE, std::move(allocation), subdomain),
    subsumed};
}

InternalSharedPtr<PhysicalStore> LogicalStore::map_reduction(
  const InternalSharedPtr<Partition>& partition,
  const SymbolicPoint& store_projection,
  const Domain& launch_domain,
  const DomainPoint& task_index,
  ReductionOpKind op_kind,
  const legate::mapping::detail::Machine& machine,
  DataMovementRequests* requests)
{
  const auto color = compute_color(store_projection, task_index);

  auto inverted = partition->invert(partition, transform_);

  auto&& [allocation, subdomain] = get_storage()->map_reduction(domain_,
                                                                inverted,
                                                                color,
                                                                machine.preferred_store_target(),
                                                                launch_domain.is_valid(),
                                                                type_,
                                                                op_kind,
                                                                requests);

  return make_internal_shared<PhysicalStore>(
    shape()->ndim(), type(), transform_, LEGION_REDUCE, std::move(allocation), subdomain);
}

InternalSharedPtr<PhysicalStore> LogicalStore::get_physical_store(
  std::optional<legate::mapping::StoreTarget> target, bool)
{
  auto&& storage = get_storage();

  const auto effective_target =
    target.value_or(Runtime::get_runtime().local_machine().has_socket_memory()
                      ? legate::mapping::StoreTarget::SOCKETMEM
                      : legate::mapping::StoreTarget::SYSMEM);

  // A physical allocation of this store is escaping to the user land for the first time,
  // so we need to make sure all outstanding operations are launched
  Runtime::get_runtime().flush_scheduling_window();

  DataMovementRequests requests{};

  auto&& [allocation, subdomain, _] =
    storage->map(domain_, create_no_partition(), {}, effective_target, /*is_read=*/true, &requests);

  // TODO(wonchanl): Handle the case when the target is FBMEM
  requests.perform_copy_ins(VariantCode::CPU);
  requests.perform_data_sync();

  return make_internal_shared<PhysicalStore>(
    shape()->ndim(), type(), transform_, LEGION_READ_WRITE, std::move(allocation), subdomain);
}

// Just because all the member functions used are const, doesn't mean this function
// is. Detaching a decidedly non-const operation.
// NOLINTNEXTLINE(readability-make-member-function-const)
void LogicalStore::detach()
{
  // TODO(wonchanl): Need a v2 implementation
}

Restrictions LogicalStore::compute_restrictions(bool is_output) const
{
  auto result = transform_->convert(get_storage()->compute_restrictions(), is_output);

  result.set_require_invertible(transformed());
  return result;
}

InternalSharedPtr<Partition> LogicalStore::find_or_create_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions)
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((num_pieces_ == new_num_pieces) && key_partition_.has_value() &&
      restrictions.are_satisfied_by(**key_partition_, shape())) {
    return *key_partition_;
  }

  if (dim() == 0 || volume() <= 1) {
    return create_no_partition();
  }

  auto storage_part = get_storage()->find_key_partition(transform_->invert(restrictions));

  if (storage_part.has_value() && (transform_->identity() || (*storage_part)->is_convertible())) {
    return (*storage_part)->convert(*storage_part, transform());
  }

  auto&& exts       = extents();
  auto&& part_mgr   = Runtime::get_runtime().partition_manager();
  auto launch_shape = part_mgr.compute_launch_shape(machine, parallel_policy, restrictions, exts);

  if (launch_shape.empty()) {
    return create_no_partition();
  }

  auto tile_shape = part_mgr.compute_tile_shape(exts, launch_shape);

  return create_tiling(std::move(tile_shape), std::move(launch_shape));
}

bool LogicalStore::has_key_partition(const mapping::detail::Machine& machine,
                                     const ParallelPolicy& parallel_policy,
                                     const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((new_num_pieces == num_pieces_) && key_partition_.has_value() &&
      restrictions.are_satisfied_by(**key_partition_, shape())) {
    return true;
  }
  return get_storage()->find_key_partition(transform_->invert(restrictions)).has_value();
}

void LogicalStore::set_key_partition(const mapping::detail::Machine& machine,
                                     const ParallelPolicy& parallel_policy,
                                     InternalSharedPtr<Partition> partition)
{
  LEGATE_ASSERT(!transformed());
  num_pieces_ = machine.count() * parallel_policy.overdecompose_factor();
  get_storage()->set_key_partition(partition);
  key_partition_ = std::move(partition);
}

void LogicalStore::reset_key_partition()
{
  key_partition_.reset();
  get_storage()->reset_key_partition();
}

InternalSharedPtr<LogicalStorePartition> LogicalStore::create_partition_(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool>)
{
  LEGATE_ASSERT(self.get() == this);
  if (unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound store cannot be manually partitioned"};
  }
  return make_internal_shared<LogicalStorePartition>(std::move(partition), self);
}

std::string LogicalStore::to_string() const
{
  auto result = fmt::format("Store({}) {{shape: ", store_id_);

  if (unbound() || deferred_bound()) {
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

  return get_storage()->equal(*other.get_storage());
}

InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
  const InternalSharedPtr<LogicalStore>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  std::optional<SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape)
{
  return self->partition_by_tiling_(self, std::move(tile_shape), std::move(color_shape));
}

InternalSharedPtr<LogicalStorePartition> create_store_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  return self->create_partition_(self, std::move(partition), std::move(complete));
}

}  // namespace legate::detail
