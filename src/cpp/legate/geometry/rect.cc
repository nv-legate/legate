/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/geometry/rect.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/zip.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <utility>

namespace legate {

Rect::iterator::iterator(Point lo_dp, Point hi_dp, bool fortran_order, bool valid)
  : lo_{std::move(lo_dp)},
    hi_{std::move(hi_dp)},
    point_{lo_},
    fortran_order_{fortran_order},
    valid_{valid}
{
  for (std::uint32_t idx = 0; valid_ && idx < lo().dim(); ++idx) {
    if (lo()[idx] > hi()[idx]) {
      valid_ = false;
      return;
    }
  }
}

void Rect::iterator::step_()
{
  const auto ndim = static_cast<std::int32_t>(point_.dim());

  if (fortran_order_) {
    // Increasing dimension order: the first dimension varies fastest.
    for (std::int32_t d = 0; d < ndim; ++d) {
      if (point_[d] < hi()[d]) {
        ++point_[d];
        return;
      }
      point_[d] = lo()[d];
    }
  } else {
    // Decreasing dimension order: the last dimension varies fastest.
    for (std::int32_t d = ndim - 1; d >= 0; --d) {
      if (point_[d] < hi()[d]) {
        ++point_[d];
        return;
      }
      point_[d] = lo()[d];
    }
  }
  // Wrapped all the way around; the range is exhausted.
  valid_ = false;
}

Rect::iterator& Rect::iterator::operator++()
{
  step_();
  return *this;
}

Rect::iterator Rect::iterator::operator++(int)
{
  auto result = *this;

  step_();
  return result;
}

bool Rect::iterator::operator==(const iterator& other) const
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

bool Rect::iterator::operator!=(const iterator& other) const { return !(*this == other); }

// ==========================================================================================

Rect::Rect(Point lo, Point hi) : lo_{std::move(lo)}, hi_{std::move(hi)}
{
  LEGATE_CHECK(lo_.dim() == hi_.dim());
}

bool Rect::operator==(const Rect& rhs) const { return lo() == rhs.lo() && hi() == rhs.hi(); }

bool Rect::operator!=(const Rect& rhs) const { return !(*this == rhs); }

Rect Rect::operator+(const Point& point) const { return Rect{lo() + point, hi() + point}; }

Rect& Rect::operator+=(const Point& point)
{
  lo_ += point;
  hi_ += point;
  return *this;
}

Rect Rect::operator-(const Point& point) const { return Rect{lo() - point, hi() - point}; }

Rect& Rect::operator-=(const Point& point)
{
  lo_ -= point;
  hi_ -= point;
  return *this;
}

bool Rect::contains(const Point& point) const
{
  LEGATE_ASSERT(point.dim() == dim());
  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    if (point[idx] < lo()[idx] || point[idx] > hi()[idx]) {
      return false;
    }
  }
  return true;
}

bool Rect::empty() const
{
  for (auto&& [l, h] : detail::zip_equal(lo(), hi())) {
    if ((h - l + 1) <= 0) {
      return true;
    }
  }
  return false;
}

std::size_t Rect::volume() const
{
  std::size_t volume = 1;

  for (auto&& [l, h] : detail::zip_equal(lo(), hi())) {
    const auto extent = h - l + 1;

    if (extent <= 0) {
      return 0;
    }
    volume *= static_cast<std::size_t>(extent);
  }
  return volume;
}

Point Rect::extents() const
{
  auto result = Point::zeroes(dim());

  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    result[idx] = std::max<std::int64_t>(0, hi()[idx] - lo()[idx] + 1);
  }
  return result;
}

Point Rect::strides() const { return extents().strides(); }

Point Rect::to_shape() const { return extents(); }

Rect Rect::intersection(const Rect& other) const
{
  LEGATE_ASSERT(dim() == other.dim());

  auto new_lo = Point::zeroes(dim());
  auto new_hi = Point::zeroes(dim());

  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    new_lo[idx] = std::max(lo()[idx], other.lo()[idx]);
    new_hi[idx] = std::min(hi()[idx], other.hi()[idx]);
  }
  return Rect{std::move(new_lo), std::move(new_hi)};
}

/*static*/ Rect Rect::from_extents(const Point& extents)
{
  return {Point::zeroes(extents.dim()), extents - std::int64_t{1}};
}

std::ostream& operator<<(std::ostream& os, const Rect& domain)
{
  os << '[' << domain.lo() << ".." << domain.hi() << ']';
  return os;
}

}  // namespace legate
