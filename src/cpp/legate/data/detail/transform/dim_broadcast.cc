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

namespace legate::detail {

Domain DimBroadcast::transform(const Domain& input) const
{
  Domain output;

  output.dim = input.dim;
  for (std::int32_t dim = 0; dim < output.dim; ++dim) {
    if (dim == dim_) {
      output.rect_data[dim]              = 0;
      output.rect_data[dim + output.dim] = static_cast<coord_t>(dim_size_) - 1;
    } else {
      output.rect_data[dim]              = input.rect_data[dim];
      output.rect_data[dim + output.dim] = input.rect_data[dim + input.dim];
    }
  }
  return output;
}

Legion::DomainAffineTransform DimBroadcast::inverse_transform(std::int32_t in_dim) const
{
  LEGATE_CHECK(dim_ < in_dim);
  const auto out_dim = in_dim;
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    for (std::int32_t j = 0; j < result.transform.n; ++j) {
      result.transform.matrix[(i * in_dim) + j] = static_cast<std::int32_t>(i == j);
    }
  }
  result.transform.matrix[(dim_ * in_dim) + dim_] = 0;

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions DimBroadcast::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  return std::move(restrictions).map([&](auto&& dim_res) {
    dim_res.at(dim_) = forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID;
    return dim_res;
  });
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  LEGATE_ASSERT(extents[dim_] == 1);
  extents.at(dim_) = dim_size_;
  return extents;
}

proj::SymbolicPoint DimBroadcast::invert(proj::SymbolicPoint point) const
{
  point.at(dim_) = constant(0);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.at(dim_) = 0;
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.at(dim_) = 1;
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> DimBroadcast::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.at(dim_) = 0;
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.at(dim_) = 1;
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
