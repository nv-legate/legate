/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition.h>

#include <legate/data/detail/logical_store.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

namespace legate::detail {

bool NoPartition::is_disjoint_for(const Domain& launch_domain) const
{
  return !launch_domain.is_valid() || launch_domain.get_volume() == 1;
}

InternalSharedPtr<Partition> NoPartition::scale(Span<const std::uint64_t> /*factors*/) const
{
  return create_no_partition();
}

InternalSharedPtr<Partition> NoPartition::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                                Span<const std::uint64_t> /*high_offsets*/) const
{
  return create_no_partition();
}

Legion::Domain NoPartition::launch_domain() const
{
  throw TracedException<std::invalid_argument>{"NoPartition has no launch domain"};
}

std::string NoPartition::to_string() const { return "NoPartition"; }

InternalSharedPtr<Partition> NoPartition::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& /*transform*/) const
{
  return self;
}

InternalSharedPtr<Partition> NoPartition::invert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& /*transform*/) const
{
  return self;
}

// ==========================================================================================

Tiling::Tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
               SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
               SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets)
  : tile_shape_{std::move(tile_shape)},
    color_shape_{std::move(color_shape)},
    offsets_{std::move(offsets)},
    strides_{tile_shape_}
{
  if (offsets_.empty()) {
    offsets_.assign(tags::size_tag, this->tile_shape().size(), 0);
  }
  LEGATE_CHECK(tile_shape_.size() == color_shape_.size());
  LEGATE_CHECK(tile_shape_.size() == offsets_.size());
}

Tiling::Tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
               SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
               SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
               SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides)
  : overlapped_{!array_all_of(std::greater_equal<>{}, strides, tile_shape)},
    tile_shape_{std::move(tile_shape)},
    color_shape_{std::move(color_shape)},
    offsets_{std::move(offsets)},
    strides_{std::move(strides)}
{
  if (this->offsets().empty()) {
    offsets_.assign(tags::size_tag, this->tile_shape().size(), 0);
  }
  LEGATE_CHECK(tile_shape_.size() == color_shape_.size());
  LEGATE_CHECK(tile_shape_.size() == offsets_.size());
}

bool Tiling::operator==(const Tiling& other) const
{
  return tile_shape_ == other.tile_shape_ && color_shape_ == other.color_shape_ &&
         offsets_ == other.offsets_ && strides_ == other.strides_;
}

bool Tiling::is_complete_for(const detail::Storage& storage) const
{
  const auto& storage_exts = storage.extents();
  const auto& storage_offs = storage.offsets();

  LEGATE_ASSERT(storage_exts.size() == storage_offs.size());
  LEGATE_ASSERT(storage_offs.size() == offsets_.size());

  const auto zip =
    legate::detail::zip_equal(offsets_, strides_, color_shape_, storage_offs, storage_exts);
  using zipper_type = std::tuple<const std::int64_t&,
                                 const std::uint64_t&,
                                 const std::uint64_t&,
                                 const std::int64_t&,
                                 const std::uint64_t&>;
  static_assert(std::is_same_v<zipper_type, decltype(*zip.begin())>);

  return std::all_of(zip.begin(), zip.end(), [](const zipper_type& zip_tuple) {
    auto&& [offset, stride, color_shape, storage_off, storage_ext] = zip_tuple;
    const auto my_lo                                               = offset;
    const auto my_hi = my_lo + static_cast<std::int64_t>(stride * color_shape);
    const auto soff  = static_cast<std::int64_t>(storage_off);

    return soff >= my_lo || my_hi >= (soff + static_cast<std::int64_t>(storage_ext));
  });
}

bool Tiling::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO(wonchanl): The check really should be that every two points from the launch domain are
  // mapped to two different colors
  return !overlapped_ &&
         (!launch_domain.is_valid() || launch_domain.get_volume() <= array_volume(color_shape_));
}

bool Tiling::satisfies_restrictions(const Restrictions& restrictions) const
{
  constexpr auto satisfies_restriction = [](Restriction r, std::uint64_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return array_all_of(satisfies_restriction, restrictions, color_shape());
}

InternalSharedPtr<Partition> Tiling::scale(Span<const std::uint64_t> factors) const
{
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> new_tile_shape;

  std::transform(tile_shape().begin(),
                 tile_shape().end(),
                 factors.begin(),
                 std::back_inserter(new_tile_shape),
                 std::multiplies<>{});

  SmallVector<std::int64_t, LEGATE_MAX_DIM> new_offsets;

  new_offsets.reserve(offsets().size());
  std::transform(
    offsets().begin(),
    offsets().end(),
    factors.begin(),
    std::back_inserter(new_offsets),
    [](std::int64_t off, std::size_t factor) { return off * static_cast<std::int64_t>(factor); });
  return create_tiling(new_tile_shape, color_shape_, new_offsets);
}

InternalSharedPtr<Partition> Tiling::bloat(Span<const std::uint64_t> low_offsets,
                                           Span<const std::uint64_t> high_offsets) const
{
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape;

  tile_shape.reserve(this->tile_shape().size());
  for (auto&& [cur_shape, lo, hi] : zip_equal(this->tile_shape(), low_offsets, high_offsets)) {
    tile_shape.emplace_back(cur_shape + lo + hi);
  }

  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets;

  offsets.reserve(this->offsets().size());
  std::transform(
    this->offsets().begin(),
    this->offsets().end(),
    low_offsets.begin(),
    std::back_inserter(offsets),
    [](std::int64_t off, std::size_t diff) { return off - static_cast<std::int64_t>(diff); });

  return create_tiling(std::move(tile_shape),
                       SmallVector<std::uint64_t, LEGATE_MAX_DIM>{color_shape()},
                       std::move(offsets),
                       SmallVector<std::uint64_t, LEGATE_MAX_DIM>{this->tile_shape()});
}

Legion::LogicalPartition Tiling::construct(Legion::LogicalRegion region, bool complete) const
{
  auto&& index_space   = region.get_index_space();
  auto&& runtime       = detail::Runtime::get_runtime();
  auto&& part_mgr      = runtime.partition_manager();
  auto index_partition = part_mgr.find_index_partition(index_space, *this);

  if (index_partition != Legion::IndexPartition::NO_PART) {
    return runtime.create_logical_partition(region, index_partition);
  }

  const auto ndim = static_cast<std::int32_t>(tile_shape_.size());
  Legion::DomainTransform transform;

  transform.m = ndim;
  transform.n = ndim;
  for (std::int32_t idx = 0; idx < ndim * ndim; ++idx) {
    transform.matrix[idx] = 0;
  }
  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    transform.matrix[(ndim * idx) + idx] = static_cast<Legion::coord_t>(strides_[idx]);
  }

  auto extent = detail::to_domain(tile_shape_);
  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    extent.rect_data[idx] += offsets_[idx];
    extent.rect_data[idx + ndim] += offsets_[idx];
  }

  auto&& color_space = runtime.find_or_create_index_space(color_shape_);
  const auto kind    = complete ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_DISJOINT_KIND;

  index_partition =
    runtime.create_restricted_partition(index_space, color_space, kind, transform, extent);
  part_mgr.record_index_partition(index_space, *this, index_partition);
  return runtime.create_logical_partition(region, index_partition);
}

Legion::Domain Tiling::launch_domain() const { return detail::to_domain(color_shape_); }

std::string Tiling::to_string() const
{
  return fmt::format("Tiling(tile: {}, colors: {}, offset: {}, strides: {})",
                     tile_shape_,
                     color_shape_,
                     offsets_,
                     strides_);
}

InternalSharedPtr<Partition> Tiling::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  return create_tiling(
    transform->convert_extents(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tile_shape()}),
    transform->convert_color_shape(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{color_shape()}),
    transform->convert_point(SmallVector<std::int64_t, LEGATE_MAX_DIM>{offsets()}),
    transform->convert_extents(strides_));
}

InternalSharedPtr<Partition> Tiling::invert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  return create_tiling(
    transform->invert_extents(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tile_shape()}),
    transform->invert_color_shape(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{color_shape()}),
    transform->invert_point(SmallVector<std::int64_t, LEGATE_MAX_DIM>{offsets()}),
    transform->invert_extents(strides_));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Tiling::get_child_extents(
  Span<const std::uint64_t> extents, Span<const std::uint64_t> color) const
{
  if (!has_color(color)) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Color {} is out of bounds, each entry must be strictly less than the "
                  "corresponding entry in {}",
                  color,
                  color_shape())};
  }

  SmallVector<std::uint64_t, LEGATE_MAX_DIM> ret;

  ret.reserve(tile_shape().size());
  for (auto&& [shape, clr, off, ext] : zip_equal(tile_shape(), color, offsets(), extents)) {
    const auto lo = std::max(std::int64_t{0}, static_cast<std::int64_t>(shape * clr) + off);
    const auto hi =
      std::min(static_cast<std::int64_t>(ext), static_cast<std::int64_t>(shape * (clr + 1)) + off);

    if (hi >= lo) {
      ret.push_back(hi - lo);
    } else {
      ret.push_back(0);
    }
  }
  return ret;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Tiling::get_child_offsets(
  Span<const std::uint64_t> color) const
{
  if (!has_color(color)) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Color {} is out of bounds, each entry must be strictly less than the "
                  "corresponding entry in {}",
                  color,
                  color_shape())};
  }

  SmallVector<std::int64_t, LEGATE_MAX_DIM> ret;

  ret.reserve(strides().size());
  for (auto&& [stride, clr, off] : zip_equal(strides(), color, offsets())) {
    const auto scaled_color = static_cast<std::int64_t>(stride * clr);

    ret.push_back(scaled_color + off);
  }
  return ret;
}

std::size_t Tiling::hash() const { return hash_all(tile_shape_, color_shape_, offsets_, strides_); }

// ==========================================================================================

Weighted::Weighted(Legion::FutureMap weights, const Domain& color_domain)
  : weights_{std::move(weights)},
    color_domain_{color_domain},
    color_shape_{detail::from_domain(color_domain)}
{
}

Weighted::~Weighted()
{
  if (detail::has_started() || !weights_.exists()) {
    return;
  }
  // FIXME: Leak the FutureMap handle if the runtime has already shut down, as there's no hope
  // that this would be collected by the Legion runtime
  static_cast<void>(std::make_unique<Legion::FutureMap>(std::move(weights_)).release());
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

bool Weighted::operator==(const Weighted& other) const
{
  // Since both color_domain_ and color_shape_ are derived from weights_, they don't need to
  // be compared
  return weights_ == other.weights_;
}

bool Weighted::operator<(const Weighted& other) const { return weights_ < other.weights_; }

bool Weighted::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO(wonchanl): The check really should be that every two points from the launch domain are
  // mapped to two different colors
  return !launch_domain.is_valid() || launch_domain.get_volume() <= color_domain_.get_volume();
}

bool Weighted::satisfies_restrictions(const Restrictions& restrictions) const
{
  constexpr auto satisfies_restriction = [](Restriction r, std::uint64_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };

  return array_all_of(satisfies_restriction, restrictions, color_shape());
}

InternalSharedPtr<Partition> Weighted::scale(Span<const std::uint64_t> /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

InternalSharedPtr<Partition> Weighted::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                             Span<const std::uint64_t> /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

Legion::LogicalPartition Weighted::construct(Legion::LogicalRegion region, bool) const
{
  auto&& runtime          = detail::Runtime::get_runtime();
  auto&& part_mgr         = runtime.partition_manager();
  const auto& index_space = region.get_index_space();
  auto index_partition    = part_mgr.find_index_partition(index_space, *this);

  if (index_partition != Legion::IndexPartition::NO_PART) {
    return runtime.create_logical_partition(region, index_partition);
  }

  auto&& color_space = runtime.find_or_create_index_space(color_shape_);

  index_partition = runtime.create_weighted_partition(index_space, color_space, weights_);
  part_mgr.record_index_partition(index_space, *this, index_partition);
  return runtime.create_logical_partition(region, index_partition);
}

std::string Weighted::to_string() const
{
  std::string result = "Weighted({";

  if (weights_.exists()) {
    for (Domain::DomainPointIterator it{color_domain_}; it; ++it) {
      auto& p = *it;

      fmt::format_to(std::back_inserter(result),
                     "{}:{},",
                     fmt::streamed(p),
                     weights_.get_result<std::size_t>(p));
    }
  }

  result += "})";
  return result;
}

InternalSharedPtr<Partition> Weighted::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
}

InternalSharedPtr<Partition> Weighted::invert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }

  const auto color_domain = [&] {
    const auto inverted =
      transform->invert_color_shape(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{color_shape()});

    return to_domain(inverted);
  }();
  // Weighted partitions are created only for 1D stores. So, if we're here, the 1D store to which
  // this partition is applied would be a degenerate N-D store such that all but one dimension are
  // of extent 1. So, we only need to delinearize the future map holding the weights so the domain
  // matches the color domain.
  return create_weighted(Runtime::get_runtime().delinearize_future_map(weights_, color_domain),
                         color_domain);
}

// ==========================================================================================

Image::Image(InternalSharedPtr<detail::LogicalStore> func,
             InternalSharedPtr<Partition> func_partition,
             mapping::detail::Machine machine,
             ImageComputationHint hint)
  : func_{std::move(func)},
    func_partition_{std::move(func_partition)},
    machine_{std::move(machine)},
    hint_{hint}
{
}

bool Image::operator==(const Image& other) const
{
  return func_->id() == other.func_->id() && func_partition_ == other.func_partition_ &&
         hint_ == other.hint_;
}

bool Image::is_disjoint_for(const Domain& launch_domain) const
{
  // Disjointedness check for image partitions is expensive, so we give a sound answer;
  return !launch_domain.is_valid();
}

bool Image::satisfies_restrictions(const Restrictions& restrictions) const
{
  constexpr auto satisfies_restriction = [](Restriction r, std::uint64_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };

  return array_all_of(satisfies_restriction, restrictions, color_shape());
}

InternalSharedPtr<Partition> Image::scale(Span<const std::uint64_t> /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

InternalSharedPtr<Partition> Image::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                          Span<const std::uint64_t> /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

Legion::LogicalPartition Image::construct(Legion::LogicalRegion region, bool /*complete*/) const
{
  if (!has_launch_domain()) {
    return Legion::LogicalPartition::NO_PART;
  }

  auto&& func_rf      = func_->get_region_field();
  auto&& func_region  = func_rf->region();
  auto func_partition = func_partition_->construct(
    func_region, func_partition_->is_complete_for(*func_->get_storage()));

  auto&& runtime  = detail::Runtime::get_runtime();
  auto&& part_mgr = runtime.partition_manager();

  auto target          = region.get_index_space();
  const auto field_id  = func_rf->field_id();
  auto index_partition = part_mgr.find_image_partition(target, func_partition, field_id, hint_);

  if (Legion::IndexPartition::NO_PART == index_partition) {
    auto construct_image_partition = [&] {
      switch (hint_) {
        case ImageComputationHint::NO_HINT: {
          const bool is_range = func_->type()->code == Type::Code::STRUCT;
          auto color_space    = runtime.find_or_create_index_space(color_shape());
          return runtime.create_image_partition(
            target, color_space, func_region, func_partition, field_id, is_range, machine_);
        }
        case ImageComputationHint::MIN_MAX: {
          return runtime.create_approximate_image_partition(func_, func_partition_, target, false);
        }
        case ImageComputationHint::FIRST_LAST: {
          return runtime.create_approximate_image_partition(func_, func_partition_, target, true);
        }
      }
      LEGATE_UNREACHABLE();
    };

    index_partition = construct_image_partition();
    part_mgr.record_image_partition(target, func_partition, field_id, hint_, index_partition);
    func_rf->add_invalidation_callback([target, func_partition, field_id, hint = hint_]() noexcept {
      detail::Runtime::get_runtime().partition_manager().invalidate_image_partition(
        target, func_partition, field_id, hint);
    });
  }

  return runtime.create_logical_partition(region, index_partition);
}

bool Image::has_launch_domain() const { return func_partition_->has_launch_domain(); }

Domain Image::launch_domain() const { return func_partition_->launch_domain(); }

std::string Image::to_string() const
{
  return fmt::format("Image(func: {}, partition: {}, hint: {})",
                     func_->to_string(),
                     func_partition_->to_string(),
                     hint_);
}

Span<const std::uint64_t> Image::color_shape() const { return func_partition_->color_shape(); }

InternalSharedPtr<Partition> Image::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
}

InternalSharedPtr<Partition> Image::invert(const InternalSharedPtr<Partition>& self,
                                           const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
}

InternalSharedPtr<NoPartition> create_no_partition()
{
  static auto result = make_internal_shared<NoPartition>();
  return result;
}

InternalSharedPtr<Tiling> create_tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
                                        SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
                                        SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets /*= {}*/)
{
  return make_internal_shared<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets));
}

InternalSharedPtr<Tiling> create_tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
                                        SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
                                        SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
                                        SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides)
{
  return make_internal_shared<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets), std::move(strides));
}

InternalSharedPtr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                            const Domain& color_domain)
{
  return make_internal_shared<Weighted>(weights, color_domain);
}

InternalSharedPtr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                      InternalSharedPtr<Partition> func_partition,
                                      mapping::detail::Machine machine,
                                      ImageComputationHint hint)
{
  return make_internal_shared<Image>(
    std::move(func), std::move(func_partition), std::move(machine), hint);
}

std::ostream& operator<<(std::ostream& out, const Partition& partition)
{
  out << partition.to_string();
  return out;
}

}  // namespace legate::detail
