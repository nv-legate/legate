/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/domain.h>
#include <legate/utilities/assert.h>

#include <utility>

namespace legate {

inline Domain::iterator::reference Domain::iterator::operator*() const { return point_; }

inline Domain::iterator::pointer Domain::iterator::operator->() const { return &point_; }

inline const DomainPoint& Domain::iterator::lo() const { return lo_; }

inline const DomainPoint& Domain::iterator::hi() const { return hi_; }

// ==========================================================================================

inline Domain::Iterable::Iterable(DomainPoint lo, DomainPoint hi, bool fortran_order)
  : lo_{std::move(lo)}, hi_{std::move(hi)}, fortran_order_{fortran_order}
{
}

inline Domain::iterator Domain::Iterable::begin() const
{
  return iterator{lo_, hi_, fortran_order_};
}

inline Domain::iterator Domain::Iterable::end() const
{
  return iterator{lo_, hi_, fortran_order_, /*valid=*/false};
}

// ==========================================================================================

template <int DIM, typename T>
Domain::Domain(const Realm::Rect<DIM, T>& rect) : lo_{rect.lo}, hi_{rect.hi}
{
}

template <int DIM, typename T>
Realm::Rect<DIM, T> Domain::bounds() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  LEGATE_ASSERT(DIM == dim());
  return Realm::Rect<DIM, T>{static_cast<Realm::Point<DIM, T>>(lo_),
                             static_cast<Realm::Point<DIM, T>>(hi_)};
}

template <int DIM, typename T>
Domain::operator Realm::Rect<DIM, T>() const
{
  return bounds<DIM, T>();
}

inline std::uint32_t Domain::dim() const { return lo_.dim(); }

inline const DomainPoint& Domain::lo() const { return lo_; }

inline const DomainPoint& Domain::hi() const { return hi_; }

inline Domain::Iterable Domain::iterate(bool fortran_order) const
{
  return Iterable{lo_, hi_, fortran_order};
}

}  // namespace legate
