/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/tiling.h>

#include <legate/data/detail/storage.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/geometry.h>
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
    strides_{tile_shape_},
    tile_shape_dp_{to_domain_point(tile_shape_)},
    color_shape_dp_{to_domain_point(color_shape_)},
    color_strides_{calculate_strides(color_shape_dp_)}
{
  if (offsets_.empty()) {
    offsets_.assign(tags::size_tag, this->tile_shape().size(), 0);
  }
  offsets_dp_ = to_domain_point(offsets_);
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
    strides_{std::move(strides)},
    tile_shape_dp_{to_domain_point(tile_shape_)},
    color_shape_dp_{to_domain_point(color_shape_)},
    color_strides_{calculate_strides(color_shape_dp_)}
{
  if (this->offsets().empty()) {
    offsets_.assign(tags::size_tag, this->tile_shape().size(), 0);
  }
  offsets_dp_ = to_domain_point(offsets_);
  LEGATE_CHECK(tile_shape_.size() == color_shape_.size());
  LEGATE_CHECK(tile_shape_.size() == offsets_.size());
}

bool Tiling::operator==(const Tiling& other) const
{
  return tile_shape_ == other.tile_shape_ && color_shape_ == other.color_shape_ &&
         offsets_ == other.offsets_ && strides_ == other.strides_;
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

Legion::Domain Tiling::launch_domain() const { return detail::to_domain(color_shape_); }

std::string Tiling::to_string() const
{
  return fmt::format("Tiling(tile: {}, colors: {}, offset: {}, strides: {}, color strides: {})",
                     tile_shape_,
                     color_shape_,
                     offsets_,
                     strides_,
                     color_strides_);
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

DomainPoint Tiling::compute_color(std::uint64_t device_id) const
{
  // TODO(wonchanl): Add device mesh
  DomainPoint result;

  result.dim = color_shape_dp_.dim;
  for (std::int32_t dim = result.dim - 1; dim > 0; --dim) {
    result[dim] = device_id % color_shape_dp_[dim];
    device_id /= color_shape_dp_[dim];
  }
  result[0] = device_id;

  return result;
}

std::uint64_t Tiling::compute_device_id(const DomainPoint& color) const
{
  // TODO(wonchanl): Add device mesh
  return inner_product(color_strides_, color);
}

Domain Tiling::shard(const Domain& domain, const DomainPoint& color) const
{
  if (!domain.is_valid()) {
    LEGATE_ASSERT(tile_shape().empty());
    return domain;
  }

  auto new_lo = color * tile_shape_dp_ + offsets_dp_;
  auto new_hi = new_lo + tile_shape_dp_;

  for (std::int32_t dim = 0; dim < new_hi.dim; ++dim) {
    new_lo[dim] = std::max(domain.rect_data[dim], new_lo[dim]);
    new_hi[dim] = std::min(new_hi[dim] - 1, domain.rect_data[dim + domain.dim]);
  }

  return {new_lo, new_hi};
}

bool Tiling::subsumes(const Domain&, const InternalSharedPtr<Partition>& other) const
{
  const auto* other_tiling = dynamic_cast<const Tiling*>(other.get());

  if (!other_tiling) {
    return false;
  }

  if (this == other_tiling) {
    return true;
  }

  // TODO(wonchanl): Isomorphism checks below can be further relaxed
  if (!std::equal(
        color_shape().begin(), color_shape().end(), other_tiling->color_shape().begin())) {
    return false;
  }

  for (auto&& [my_tile_ext, color_ext, my_off, other_tile_ext, other_off] :
       zip_equal(tile_shape(),
                 color_shape(),
                 offsets(),
                 other_tiling->tile_shape(),
                 other_tiling->offsets())) {
    if (my_tile_ext == other_tile_ext && my_off == other_off) {
      continue;
    }
    if (color_ext > 1) {
      return false;
    }
    // If this dimension isn't partitioned and the tile extents don't match, we can at least check
    // if the other tile is smaller than my tile for this dimension.
    if (my_tile_ext + my_off < other_tile_ext + other_off) {
      return false;
    }
  }

  return true;
}

std::size_t Tiling::hash() const { return hash_all(tile_shape_, color_shape_, offsets_, strides_); }

// ==========================================================================================

InternalSharedPtr<Partition> create_tiling(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets /*= {}*/)
{
  if (tile_shape.empty()) {
    return create_no_partition();
  }
  return make_internal_shared<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets));
}

InternalSharedPtr<Partition> create_tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
                                           SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
                                           SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
                                           SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides)
{
  if (tile_shape.empty()) {
    return create_no_partition();
  }
  return make_internal_shared<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets), std::move(strides));
}

}  // namespace legate::detail
