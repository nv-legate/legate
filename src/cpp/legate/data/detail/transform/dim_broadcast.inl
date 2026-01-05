/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/dim_broadcast.h>

namespace legate::detail {

inline DimBroadcast::DimBroadcast(std::int32_t dim, std::uint64_t dim_size)
  : dim_{dim}, dim_size_{dim_size}
{
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return color;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return color_shape;
}

inline SmallVector<std::int64_t, LEGATE_MAX_DIM> DimBroadcast::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  return point;
}

inline Restrictions DimBroadcast::invert(Restrictions restrictions) const { return restrictions; }

inline bool DimBroadcast::is_convertible() const { return true; }

inline std::int32_t DimBroadcast::target_ndim(std::int32_t source_ndim) const
{
  return source_ndim;
}

inline SmallVector<std::int32_t, LEGATE_MAX_DIM> DimBroadcast::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  return dims;
}

}  // namespace legate::detail
