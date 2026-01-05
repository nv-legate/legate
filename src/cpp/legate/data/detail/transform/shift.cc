/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/shift.h>

#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iostream>

namespace legate::detail {

Domain Shift::transform(const Domain& input) const
{
  auto result = input;
  result.rect_data[dim_] += offset_;
  result.rect_data[dim_ + result.dim] += offset_;
  return result;
}

Legion::DomainAffineTransform Shift::inverse_transform(std::int32_t in_dim) const
{
  LEGATE_CHECK(dim_ < in_dim);
  const auto out_dim = in_dim;
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    for (std::int32_t j = 0; j < in_dim; ++j) {
      result.transform.matrix[(i * in_dim) + j] = static_cast<coord_t>(i == j);
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = i == dim_ ? -offset_ : 0;
  }
  return result;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Shift::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point[dim_] += offset_;
  return point;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Shift::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point[dim_] -= offset_;
  return point;
}

void Shift::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::SHIFT);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::int64_t>(offset_);
}

void Shift::print(std::ostream& out) const
{
  out << "Shift(dim: " << dim_ << ", "
      << "offset: " << offset_ << ")";
}

SmallVector<std::int32_t, LEGATE_MAX_DIM> Shift::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  return dims;
}

}  // namespace legate::detail
