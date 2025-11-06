/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/transpose.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <utility>

namespace legate::detail {

Transpose::Transpose(SmallVector<std::int32_t, LEGATE_MAX_DIM>&& axes) : axes_{std::move(axes)}
{
  const auto size = axes_.size();
  // could alternatively do
  //
  // inverse_.resize(size);
  // std::iota(inverse_.begin(), inverse_.end(), 0);
  //
  // but this results in 2 traversals of the array, once to initialize inverse_ to 0, and a
  // second time to do the iota-ing
  inverse_.reserve(size);
  std::generate_n(std::back_inserter(inverse_), size, [n = 0]() mutable { return n++; });
  std::sort(inverse_.begin(), inverse_.end(), [&](std::int32_t idx1, std::int32_t idx2) {
    return axes_[static_cast<std::size_t>(idx1)] < axes_[static_cast<std::size_t>(idx2)];
  });
}

Domain Transpose::transform(const Domain& domain) const
{
  Domain output;

  output.dim = domain.dim;
  for (std::int32_t out_dim = 0; out_dim < output.dim; ++out_dim) {
    auto in_dim                            = axes_[out_dim];
    output.rect_data[out_dim]              = domain.rect_data[in_dim];
    output.rect_data[out_dim + output.dim] = domain.rect_data[in_dim + domain.dim];
  }
  return output;
}

Legion::DomainAffineTransform Transpose::inverse_transform(std::int32_t in_dim) const
{
  Legion::DomainAffineTransform result;

  result.transform.m = in_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < in_dim; ++i) {
    for (std::int32_t j = 0; j < in_dim; ++j) {
      result.transform.matrix[(i * in_dim) + j] = 0;
    }
  }

  for (std::int32_t j = 0; j < in_dim; ++j) {
    result.transform.matrix[(axes_[j] * in_dim) + j] = 1;
  }

  result.offset.dim = in_dim;
  for (std::int32_t i = 0; i < in_dim; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions Transpose::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  // No in-place available
  return array_map(restrictions, axes_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  // No in-place available
  return array_map(color, axes_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  // No in-place available
  return array_map(color_shape, axes_);
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Transpose::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  // No in-place available
  return array_map(point, axes_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  // No in-place available
  return array_map(extents, axes_);
}

proj::SymbolicPoint Transpose::invert(proj::SymbolicPoint point) const
{
  // No in-place available
  return point.map(inverse_);
}

Restrictions Transpose::invert(Restrictions restrictions) const
{
  // No in-place available
  return array_map(restrictions, inverse_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  // No in-place available
  return array_map(color, inverse_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  // No in-place available
  return array_map(color_shape, inverse_);
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Transpose::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  // No in-place available
  return array_map(point, inverse_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  // No in-place available
  return array_map(extents, inverse_);
}

void Transpose::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::TRANSPOSE);
  buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(axes_.size()));
  for (auto axis : axes_) {
    buffer.pack<std::int32_t>(axis);
  }
}

void Transpose::print(std::ostream& out) const
{
  out << fmt::format("Transpose(axes: {})", fmt::join(axes_, ", "));
}

void Transpose::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  // i should be added to X.transpose(axes).promoted iff axes[i] is in X.promoted
  // e.g. X.promoted = [0] => X.transpose((1,2,0)).promoted = [2]
  for (auto&& promoted : dims) {
    auto finder = std::find(axes_.begin(), axes_.end(), promoted);

    LEGATE_CHECK(finder != axes_.end());
    promoted = static_cast<std::int32_t>(finder - axes_.begin());
  }
}

SmallVector<std::int32_t, LEGATE_MAX_DIM> Transpose::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  decltype(dims) ret;

  ret.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(ret), [&](std::int32_t dim) {
    // Unlike points or shapes, whose indices correspond to dimension indices, `dims` has dimension
    // indices as values, so the inversion is simply looking up the `axes_` using those values.
    return axes_[dim];
  });
  return ret;
}

}  // namespace legate::detail
