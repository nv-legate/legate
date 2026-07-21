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
inline SymbolicPoint Shift::invert(SymbolicPoint point) const { return point; }

inline Restrictions Shift::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  return restrictions;
}

inline Point Shift::convert_color(Point color) const { return color; }

inline Point Shift::convert_color_shape(Point color_shape) const { return color_shape; }

inline Point Shift::convert_extents(Point extents) const { return extents; }

// A shift only offsets coordinates; it does not change the storage layout, so the strides are
// unchanged (the offset is folded into the base pointer via `invert_point`).
inline Point Shift::convert_strides(Point strides) const { return strides; }

inline Restrictions Shift::invert(Restrictions restrictions) const { return restrictions; }

inline Point Shift::invert_color(Point color) const { return color; }

inline Point Shift::invert_color_shape(Point color_shape) const { return color_shape; }

inline Point Shift::invert_extents(Point extents) const { return extents; }

inline std::int32_t Shift::target_ndim(std::int32_t source_ndim) const { return source_ndim; }

inline void Shift::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const {}

}  // namespace legate::detail
