/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store.h>

#include <legate_defines.h>

#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/transform.h>
#include <legate/mapping/detail/machine.h>
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
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/small_vector.h>
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

LogicalStore::LogicalStore(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
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

  auto new_extents = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{extents()};

  new_extents.insert(new_extents.begin() + extra_dim, dim_size);

  auto transform = stack(transform_, std::make_unique<Promote>(extra_dim, dim_size));

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

  auto new_extents = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{old_extents};

  new_extents.erase(new_extents.begin() + d);

  auto transform = stack(transform_, std::make_unique<Project>(d, index));
  auto substorage =
    volume() == 0 ? storage_
                  : slice_storage(storage_,
                                  transform->invert_extents(new_extents),
                                  transform->invert_point({tags::size_tag, new_extents.size(), 0}));

  return make_internal_shared<LogicalStore>(
    std::move(new_extents), std::move(substorage), type(), std::move(transform));
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
    std::move(new_extents), storage_, type(), std::move(transform));
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

  if (0 == exts[dim]) {
    return Runtime::get_runtime().create_store(
      make_internal_shared<Shape>(std::move(exts)), type(), false /*optimize_scalar*/);
  }

  auto transform =
    (start == 0) ? transform_ : stack(transform_, std::make_unique<Shift>(dim, -start));
  auto substorage = volume() == 0
                      ? storage_
                      : slice_storage(storage_,
                                      transform->invert_extents(exts),
                                      transform->invert_point({tags::size_tag, exts.size(), 0}));
  return make_internal_shared<LogicalStore>(
    std::move(exts), std::move(substorage), type(), std::move(transform));
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
    std::move(new_extents), storage_, type(), std::move(transform));
}

InternalSharedPtr<LogicalStore> LogicalStore::delinearize(
  std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes)
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
           std::all_of(begin, end, [&to_match](auto size) { return size <= to_match; });
  };

  if (!delinearizable(old_extents[dim], sizes)) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Dimension of size {} cannot be delinearized into {}", old_extents[dim], sizes)};
  }

  auto new_extents = SmallVector<std::uint64_t, LEGATE_MAX_DIM>{};

  new_extents.reserve(old_extents.size() + sizes.size());
  std::copy_n(old_extents.begin(), dim, std::back_inserter(new_extents));
  std::copy(sizes.begin(), sizes.end(), std::back_inserter(new_extents));
  if (static_cast<std::uint32_t>(dim + 1) < old_extents.size()) {
    std::copy(old_extents.begin() + dim + 1, old_extents.end(), std::back_inserter(new_extents));
  }

  auto transform = stack(transform_, std::make_unique<Delinearize>(dim, std::move(sizes)));

  return make_internal_shared<LogicalStore>(
    std::move(new_extents), storage_, type(), std::move(transform));
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
  Span<const std::uint64_t> color_shape,
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

void LogicalStore::maybe_reset_key_partition_(const Partition* to_match) noexcept
{
  if (!key_partition_.has_value() || key_partition_->get() != to_match) {
    return;
  }
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

StoreAnalyzable LogicalStore::to_launcher_arg_(const InternalSharedPtr<LogicalStore>& self,
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

  LEGATE_ABORT("Unhandled storage kind ", to_underlying(get_storage()->kind()));
}

StoreAnalyzable LogicalStore::future_to_launcher_arg_(Legion::Future future,
                                                      const Domain& launch_domain,
                                                      Legion::PrivilegeMode privilege,
                                                      GlobalRedopID redop)
{
  // The code below has not been updated to take other flags into account.
  LEGATE_CHECK(privilege == LEGION_REDUCE || privilege == LEGION_WRITE_ONLY ||
               privilege == LEGION_READ_ONLY);
  if (!launch_domain.is_valid() && LEGION_REDUCE == privilege) {
    privilege = LEGION_READ_WRITE;
  }

  if (!future.exists()) {
    if (privilege != LEGION_WRITE_ONLY) {
      throw TracedException<std::invalid_argument>{
        "Read access or reduction to an uninitialized scalar store is prohibited"};
    }
    return WriteOnlyScalarStoreArg{this, redop};
  }

  // Scalar reductions don't need to pass the future or future map holding the current value to the
  // task, as the physical stores will be initialized with the reduction identity. They are later
  // passed to a future map reduction as an initial value in the task launch postamble.
  if (privilege == LEGION_REDUCE) {
    return WriteOnlyScalarStoreArg{this, redop};
  }

  // TODO(wonchanl): technically, we can create a WriteOnlyScalarStoreArg when privilege is
  // LEGION_WRITE_ONLY. Unfortunately, we don't currently track scalar stores passed as both inputs
  // and outputs, which are currently mapped to separate physical stores in the task. So, the
  // privilege of this store alone doesn't tell us whether it's truly a write-only store or this is
  // also passed as an input store. For the time being, we just pass the future when it exists even
  // when the store is not actually read by the task.
  return ScalarStoreArg{
    this, std::move(future), get_storage()->scalar_offset(), privilege == LEGION_READ_ONLY, redop};
}

StoreAnalyzable LogicalStore::future_map_to_launcher_arg_(const Domain& launch_domain,
                                                          Legion::PrivilegeMode privilege,
                                                          GlobalRedopID redop)
{
  if (unbound()) {
    return WriteOnlyScalarStoreArg{this, GlobalRedopID{-1} /*redop*/};
  }
  LEGATE_ASSERT(get_storage()->replicated());

  auto future_or_future_map = get_storage()->get_future_or_future_map(launch_domain);

  return std::visit(
    Overload{
      [&](Legion::Future fut) {
        return future_to_launcher_arg_(std::move(fut), launch_domain, privilege, redop);
      },
      [&](Legion::FutureMap map) -> StoreAnalyzable {
        // The code below has not been updated to take other flags into account.
        LEGATE_CHECK(privilege == LEGION_REDUCE || privilege == LEGION_WRITE_ONLY ||
                     privilege == LEGION_READ_ONLY);
        // Scalar reductions don't need to pass the future or future map holding the current value
        // to the task, as the physical stores will be initialized with the reduction identity. They
        // are later passed to a future map reduction as an initial value in the task launch
        // postamble.
        if (privilege == LEGION_REDUCE) {
          return WriteOnlyScalarStoreArg{this, redop};
        }
        // TODO(wonchanl): technically, we can create a WriteOnlyScalarStoreArg when privilege is
        // LEGION_WRITE_ONLY. Unfortunately, we don't currently track scalar stores passed as both
        // inputs and outputs, which are currently mapped to separate physical stores in the task.
        // So, the privilege of this store alone doesn't tell us whether it's truly a write-only
        // store or this is also passed as an input store. For the time being, we just pass the
        // future when it exists even when the store is not actually read by the task.
        return ReplicatedScalarStoreArg{
          this, std::move(map), get_storage()->scalar_offset(), privilege == LEGION_READ_ONLY};
      }},
    future_or_future_map);
}

StoreAnalyzable LogicalStore::region_field_to_launcher_arg_(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  GlobalRedopID redop)
{
  if (unbound()) {
    auto&& [field_space, field_id] = strategy.find_field_for_unbound_store(*variable);
    return OutputRegionArg{this, field_space, field_id};
  }

  auto&& partition     = strategy[*variable];
  auto store_partition = create_partition_(self, partition);
  auto store_proj      = store_partition->create_store_projection(launch_domain, projection);

  store_proj.is_key = strategy.is_key_partition(*variable);
  store_proj.redop  = redop;

  // We ignore LEGIION_DISCARD_OUTPUT_MASK below because it is "fake". This privilege is
  // artificially added during streaming sections to allow early discards of the
  // operands. Otherwise, operands should only have either:
  //
  // 1. LEGION_READ_ONLY.
  // 2. LEGION_WRITE_ONLY.
  // 3. LEGION_REDUCE.
  if ((ignore_privilege(privilege, LEGION_DISCARD_OUTPUT_MASK) == LEGION_REDUCE) &&
      store_partition->is_disjoint_for(launch_domain)) {
    privilege |= LEGION_READ_WRITE;
  }
  if (((ignore_privilege(privilege, LEGION_DISCARD_OUTPUT_MASK) == LEGION_WRITE_ONLY) ||
       (ignore_privilege(privilege, LEGION_DISCARD_OUTPUT_MASK) == LEGION_READ_WRITE)) &&
      partition->has_launch_domain()) {
    const auto* op = variable->operation();

    set_key_partition(op->machine(), op->parallel_policy(), partition);

    // If the cached key partition is an image partition, we need to make sure that the cache
    // doesn't outlive the store used to derive the partition (otherwise the image operation sees an
    // uninitialized store).
    if (auto* const image = dynamic_cast<Image*>(partition.get()); image) {
      image->func()->get_region_field()->add_invalidation_callback(
        [weak_self = InternalWeakPtr<LogicalStore>{self}, to_match = partition.get()]() noexcept {
          if (auto maybe_self = weak_self.lock(); maybe_self) {
            maybe_self->maybe_reset_key_partition_(to_match);
          }
        });
    }
  }

  return RegionFieldArg{this, privilege, std::move(store_proj)};
}

RegionFieldArg LogicalStore::to_launcher_arg_for_fixup_(const InternalSharedPtr<LogicalStore>& self,
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
  return RegionFieldArg{this, privilege, std::move(store_proj)};
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
