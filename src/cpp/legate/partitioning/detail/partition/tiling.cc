/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/tiling.h>

#include <legate/data/detail/storage.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace legate::detail {

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

bool Tiling::has_color(Span<const std::uint64_t> color) const
{
  return std::equal(
    color.begin(), color.end(), color_shape().begin(), color_shape().end(), std::less<>{});
}

// ==========================================================================================

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

}  // namespace legate::detail
