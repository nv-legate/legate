/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/nd_rect.h>

#include <ostream>

namespace legate {

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM>::iterator::iterator(const NDPoint<DIM>& lo,
                                                   const NDPoint<DIM>& hi,
                                                   bool fortran_order,
                                                   bool valid)
  : point_{lo}, lo_{lo}, hi_{hi}, fortran_order_{fortran_order}, valid_{valid}
{
  for (int idx = 0; valid_ && idx < DIM; ++idx) {
    if (lo_[idx] > hi_[idx]) {
      valid_ = false;
      break;
    }
  }
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::iterator::reference NDRect<DIM>::iterator::operator*()
  const
{
  return point_;
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::iterator::pointer NDRect<DIM>::iterator::operator->() const
{
  return &point_;
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::iterator& NDRect<DIM>::iterator::operator++()
{
  step_();
  return *this;
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::iterator NDRect<DIM>::iterator::operator++(int)
{
  auto result = *this;

  step_();
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE bool NDRect<DIM>::iterator::operator==(const iterator& other) const
{
  if (valid_ != other.valid_) {
    return false;
  }
  if (!valid_) {
    // Both iterators are past-the-end.
    return true;
  }
  return point_ == other.point_;
}

template <int DIM>
LEGATE_HOST_DEVICE bool NDRect<DIM>::iterator::operator!=(const iterator& other) const
{
  return !(*this == other);
}

template <int DIM>
LEGATE_HOST_DEVICE void NDRect<DIM>::iterator::step_()
{
  if (fortran_order_) {
    // Increasing dimension order: the first dimension varies fastest.
    for (int d = 0; d < DIM; ++d) {
      if (point_[d] < hi_[d]) {
        ++point_[d];
        return;
      }
      point_[d] = lo_[d];
    }
  } else {
    // Decreasing dimension order: the last dimension varies fastest.
    for (int d = DIM - 1; d >= 0; --d) {
      if (point_[d] < hi_[d]) {
        ++point_[d];
        return;
      }
      point_[d] = lo_[d];
    }
  }
  // Wrapped all the way around; the range is exhausted.
  valid_ = false;
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM>::Iterable::Iterable(const NDPoint<DIM>& lo,
                                                   const NDPoint<DIM>& hi,
                                                   bool fortran_order)
  : lo_{lo}, hi_{hi}, fortran_order_{fortran_order}
{
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::iterator NDRect<DIM>::Iterable::begin() const
{
  return iterator{lo_, hi_, fortran_order_, /*valid=*/true};
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::iterator NDRect<DIM>::Iterable::end() const
{
  return iterator{lo_, hi_, fortran_order_, /*valid=*/false};
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM>::NDRect(const NDPoint<DIM>& lo_, const NDPoint<DIM>& hi_)
  : lo{lo_}, hi{hi_}
{
}

template <int DIM>
LEGATE_HOST_DEVICE bool NDRect<DIM>::empty() const
{
  for (int idx = 0; idx < DIM; ++idx) {
    if (lo[idx] > hi[idx]) {
      return true;
    }
  }
  return false;
}

template <int DIM>
LEGATE_HOST_DEVICE std::size_t NDRect<DIM>::volume() const
{
  std::size_t result = 1;

  for (int idx = 0; idx < DIM; ++idx) {
    if (lo[idx] > hi[idx]) {
      return 0;
    }
    result *= (static_cast<std::size_t>(hi[idx]) - static_cast<std::size_t>(lo[idx]) + 1);
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE bool NDRect<DIM>::contains(const NDPoint<DIM>& point) const
{
  for (int idx = 0; idx < DIM; ++idx) {
    if (point[idx] < lo[idx] || point[idx] > hi[idx]) {
      return false;
    }
  }
  return true;
}

template <int DIM>
LEGATE_HOST_DEVICE bool NDRect<DIM>::contains(const NDRect<DIM>& other) const
{
  bool result = true;

  for (int idx = 0; idx < DIM; ++idx) {
    if (other.lo[idx] > other.hi[idx]) {
      // `other` is empty in this dimension, hence empty overall: trivially contained.
      return true;
    }
    if (lo[idx] > other.lo[idx] || other.hi[idx] > hi[idx]) {
      result = false;
    }
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE bool NDRect<DIM>::overlaps(const NDRect<DIM>& other) const
{
  for (int idx = 0; idx < DIM; ++idx) {
    if (lo[idx] > hi[idx] || lo[idx] > other.hi[idx] || other.lo[idx] > hi[idx] ||
        other.lo[idx] > other.hi[idx]) {
      return false;
    }
  }
  return true;
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM> NDRect<DIM>::intersection(const NDRect<DIM>& other) const
{
  NDRect<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result.lo[idx] = lo[idx] < other.lo[idx] ? other.lo[idx] : lo[idx];  // max
    result.hi[idx] = hi[idx] < other.hi[idx] ? hi[idx] : other.hi[idx];  // min
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM> NDRect<DIM>::union_bbox(const NDRect<DIM>& other) const
{
  if (empty()) {
    return other;
  }
  if (other.empty()) {
    return *this;
  }

  NDRect<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result.lo[idx] = lo[idx] < other.lo[idx] ? lo[idx] : other.lo[idx];  // min
    result.hi[idx] = hi[idx] < other.hi[idx] ? other.hi[idx] : hi[idx];  // max
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDRect<DIM>::Iterable NDRect<DIM>::iterate(bool fortran_order) const
{
  return Iterable{lo, hi, fortran_order};
}

template <int DIM>
LEGATE_HOST_DEVICE bool operator==(const NDRect<DIM>& lhs, const NDRect<DIM>& rhs)
{
  return lhs.lo == rhs.lo && lhs.hi == rhs.hi;
}

template <int DIM>
LEGATE_HOST_DEVICE bool operator!=(const NDRect<DIM>& lhs, const NDRect<DIM>& rhs)
{
  return !(lhs == rhs);
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM> operator+(const NDRect<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  return NDRect<DIM>{lhs.lo + rhs, lhs.hi + rhs};
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM>& operator+=(NDRect<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  lhs.lo += rhs;
  lhs.hi += rhs;
  return lhs;
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM> operator-(const NDRect<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  return NDRect<DIM>{lhs.lo - rhs, lhs.hi - rhs};
}

template <int DIM>
LEGATE_HOST_DEVICE NDRect<DIM>& operator-=(NDRect<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  lhs.lo -= rhs;
  lhs.hi -= rhs;
  return lhs;
}

template <int DIM>
std::ostream& operator<<(std::ostream& os, const NDRect<DIM>& rect)
{
  os << rect.lo << ".." << rect.hi;
  return os;
}

}  // namespace legate
