/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/rect.h>
#include <legate/utilities/assert.h>

#include <utility>

namespace legate {

inline Rect::iterator::reference Rect::iterator::operator*() const { return point_; }

inline Rect::iterator::pointer Rect::iterator::operator->() const { return &point_; }

inline const Point& Rect::iterator::lo() const { return lo_; }

inline const Point& Rect::iterator::hi() const { return hi_; }

// ==========================================================================================

inline Rect::Iterable::Iterable(Point lo, Point hi, bool fortran_order)
  : lo_{std::move(lo)}, hi_{std::move(hi)}, fortran_order_{fortran_order}
{
}

inline Rect::iterator Rect::Iterable::begin() const { return iterator{lo_, hi_, fortran_order_}; }

inline Rect::iterator Rect::Iterable::end() const
{
  return iterator{lo_, hi_, fortran_order_, /*valid=*/false};
}

// ==========================================================================================

template <int DIM>
Rect::Rect(const NDRect<DIM>& rect) : lo_{rect.lo}, hi_{rect.hi}
{
}

template <int DIM>
NDRect<DIM> Rect::bounds() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  LEGATE_ASSERT(DIM == dim());
  return NDRect<DIM>{static_cast<NDPoint<DIM>>(lo_), static_cast<NDPoint<DIM>>(hi_)};
}

template <int DIM>
Rect::operator NDRect<DIM>() const
{
  return bounds<DIM>();
}

inline std::uint32_t Rect::dim() const { return lo_.dim(); }

inline const Point& Rect::lo() const { return lo_; }

inline Point& Rect::lo() { return lo_; }

inline const Point& Rect::hi() const { return hi_; }

inline Point& Rect::hi() { return hi_; }

inline Rect::Iterable Rect::iterate(bool fortran_order) const
{
  return Iterable{lo_, hi_, fortran_order};
}

}  // namespace legate
