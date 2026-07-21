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
#include <utility>

namespace legate::detail {

Rect Shift::transform(const Rect& input) const
{
  auto lo = input.lo();
  auto hi = input.hi();

  lo[dim_] += offset_;
  hi[dim_] += offset_;
  return {std::move(lo), std::move(hi)};
}

Point Shift::convert_point(Point point) const
{
  point[dim_] += offset_;
  return point;
}

Point Shift::invert_point(Point point) const
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
