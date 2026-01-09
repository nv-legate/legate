/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/shift.h>

namespace legate::detail {

inline Shift::Shift(std::int32_t dim, std::int64_t offset) : dim_{dim}, offset_{offset} {}

// the shift transform makes no change on the store's dimensions
inline proj::SymbolicPoint Shift::invert(proj::SymbolicPoint point) const { return point; }

inline Restrictions Shift::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  return restrictions;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return color;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return color_shape;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  return extents;
}

inline Restrictions Shift::invert(Restrictions restrictions) const { return restrictions; }

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return color;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return color_shape;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  return extents;
}

inline std::int32_t Shift::target_ndim(std::int32_t source_ndim) const { return source_ndim; }

inline void Shift::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const {}

inline bool Shift::is_convertible() const { return true; }

}  // namespace legate::detail
