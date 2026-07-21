/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/dim_broadcast.h>

#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <cstdint>
#include <iostream>
#include <utility>

namespace legate::detail {

Domain DimBroadcast::transform(const Domain& input) const
{
  const auto ndim = input.dim();
  auto lo         = DomainPoint::zeroes(ndim);
  auto hi         = DomainPoint::zeroes(ndim);
  auto&& in_lo    = input.lo();
  auto&& in_hi    = input.hi();

  for (std::uint32_t dim = 0; dim < ndim; ++dim) {
    if (static_cast<std::int32_t>(dim) == dim_) {
      lo[dim] = 0;
      hi[dim] = static_cast<std::int64_t>(dim_size_) - 1;
    } else {
      lo[dim] = in_lo[dim];
      hi[dim] = in_hi[dim];
    }
  }
  return {std::move(lo), std::move(hi)};
}

Restrictions DimBroadcast::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  return std::move(restrictions)
    .map(
      [&](auto&& dim_res) {
        dim_res.at(dim_) = forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID;
        return dim_res;
      },
      [&](auto&& min_exts) { return min_exts; });
}

DomainPoint DimBroadcast::convert_extents(DomainPoint extents) const
{
  LEGATE_ASSERT(extents[dim_] == 1);
  extents[dim_] = static_cast<std::int64_t>(dim_size_);
  return extents;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::convert_strides(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides) const
{
  // Broadcasting reads the same (unit-extent) backing element for every index along `dim_`, so
  // its stride must be 0.
  strides.at(dim_) = 0;
  return strides;
}

SymbolicPoint DimBroadcast::invert(SymbolicPoint point) const
{
  point.at(dim_) = constant(0);
  return point;
}

DomainPoint DimBroadcast::invert_color(DomainPoint color) const
{
  color[dim_] = 0;
  return color;
}

DomainPoint DimBroadcast::invert_color_shape(DomainPoint color_shape) const
{
  color_shape[dim_] = 1;
  return color_shape;
}

DomainPoint DimBroadcast::invert_point(DomainPoint point) const
{
  point[dim_] = 0;
  return point;
}

DomainPoint DimBroadcast::invert_extents(DomainPoint extents) const
{
  extents[dim_] = 1;
  return extents;
}

void DimBroadcast::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::BROADCAST);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::uint64_t>(dim_size_);
}

void DimBroadcast::print(std::ostream& out) const
{
  out << fmt::format("Broadcast(dim: {}, dim_size: {})", dim_, dim_size_);
}

void DimBroadcast::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  dims.push_back(dim_);
}

}  // namespace legate::detail
