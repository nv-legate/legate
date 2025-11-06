/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/promote.h>

#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace legate::detail {

Domain Promote::transform(const Domain& input) const
{
  Domain output;

  output.dim = input.dim + 1;
  for (std::int32_t out_dim = 0, in_dim = 0; out_dim < output.dim; ++out_dim) {
    if (out_dim == extra_dim_) {
      output.rect_data[out_dim]              = 0;
      output.rect_data[out_dim + output.dim] = dim_size_ - 1;
    } else {
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
      ++in_dim;
    }
  }
  return output;
}

Legion::DomainAffineTransform Promote::inverse_transform(std::int32_t in_dim) const
{
  LEGATE_CHECK(extra_dim_ < in_dim);
  const auto out_dim = in_dim - 1;
  Legion::DomainAffineTransform result;

  result.transform.m = std::max<std::int32_t>(out_dim, 1);
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    for (std::int32_t j = 0; j < result.transform.n; ++j) {
      result.transform.matrix[(i * in_dim) + j] = 0;
    }
  }

  if (out_dim > 0) {
    for (std::int32_t j = 0, i = 0; j < result.transform.n; ++j) {
      if (j != extra_dim_) {
        result.transform.matrix[(i++ * in_dim) + j] = 1;
      }
    }
  }

  result.offset.dim = std::max<std::int32_t>(out_dim, 1);
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions Promote::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  restrictions.insert(restrictions.begin() + extra_dim_,
                      forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID);
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.insert(color.begin() + extra_dim_, 0);
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.insert(color_shape.begin() + extra_dim_, 1);
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Promote::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.insert(point.begin() + extra_dim_, 0);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.insert(extents.begin() + extra_dim_, dim_size_);
  return extents;
}

proj::SymbolicPoint Promote::invert(proj::SymbolicPoint point) const
{
  point.remove_inplace(extra_dim_);
  return point;
}

Restrictions Promote::invert(Restrictions restrictions) const
{
  restrictions.erase(restrictions.begin() + extra_dim_);
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.erase(color.begin() + extra_dim_);
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.erase(color_shape.begin() + extra_dim_);
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Promote::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.erase(point.begin() + extra_dim_);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.erase(extents.begin() + extra_dim_);
  return extents;
}

void Promote::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::PROMOTE);
  buffer.pack<std::int32_t>(extra_dim_);
  buffer.pack<std::int64_t>(dim_size_);
}

void Promote::print(std::ostream& out) const
{
  out << fmt::format("Promote(extra_dim: {}, dim_size: {})", extra_dim_, dim_size_);
}

void Promote::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  for (auto&& dim : dims) {
    if (dim >= extra_dim_) {
      dim++;
    }
  }
  dims.push_back(extra_dim_);
}

SmallVector<std::int32_t, LEGATE_MAX_DIM> Promote::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  // Remove the promoted dimension from the ordering(extra_dim_)
  // and renumber dims > extra_dim_

  const auto it = std::find(dims.begin(), dims.end(), extra_dim_);

  // If the dimension added by this promotion is projected in the transformation stack,
  // the dimension index does not exist in `dims`.
  if (it != dims.end()) {
    dims.erase(it);
  }

  std::transform(
    dims.begin(), dims.end(), dims.begin(), [&](auto d) { return d > extra_dim_ ? d - 1 : d; });

  return dims;
}

}  // namespace legate::detail
