/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/geometry/domain.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/zip.h>

#include <algorithm>
#include <cstddef>
#include <ostream>
#include <utility>

namespace legate {

Domain::iterator::iterator(std::reference_wrapper<const DomainPoint> lo_dp,
                           std::reference_wrapper<const DomainPoint> hi_dp,
                           bool fortran_order,
                           bool valid)
  : lo_{std::move(lo_dp)},
    hi_{std::move(hi_dp)},
    point_{lo()},
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

void Domain::iterator::step_()
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

Domain::iterator& Domain::iterator::operator++()
{
  step_();
  return *this;
}

Domain::iterator Domain::iterator::operator++(int)
{
  auto result = *this;

  step_();
  return result;
}

bool Domain::iterator::operator==(const iterator& other) const
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

bool Domain::iterator::operator!=(const iterator& other) const { return !(*this == other); }

// ==========================================================================================

Domain::Domain(DomainPoint lo, DomainPoint hi) : lo_{std::move(lo)}, hi_{std::move(hi)}
{
  LEGATE_CHECK(lo_.dim() == hi_.dim());
}

bool Domain::operator==(const Domain& rhs) const { return lo() == rhs.lo() && hi() == rhs.hi(); }

bool Domain::operator!=(const Domain& rhs) const { return !(*this == rhs); }

Domain Domain::operator+(const DomainPoint& point) const
{
  return Domain{lo() + point, hi() + point};
}

Domain& Domain::operator+=(const DomainPoint& point)
{
  lo_ += point;
  hi_ += point;
  return *this;
}

Domain Domain::operator-(const DomainPoint& point) const
{
  return Domain{lo() - point, hi() - point};
}

Domain& Domain::operator-=(const DomainPoint& point)
{
  lo_ -= point;
  hi_ -= point;
  return *this;
}

bool Domain::contains(const DomainPoint& point) const
{
  LEGATE_ASSERT(point.dim() == dim());
  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    if (point[idx] < lo()[idx] || point[idx] > hi()[idx]) {
      return false;
    }
  }
  return true;
}

bool Domain::empty() const
{
  for (auto&& [l, h] : detail::zip_equal(lo(), hi())) {
    if ((h - l + 1) <= 0) {
      return true;
    }
  }
  return false;
}

std::size_t Domain::volume() const
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

DomainPoint Domain::extents() const
{
  auto result = DomainPoint::zeroes(dim());

  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    result[idx] = std::max<std::int64_t>(0, hi()[idx] - lo()[idx] + 1);
  }
  return result;
}

DomainPoint Domain::strides() const { return extents().strides(); }

DomainPoint Domain::to_shape() const { return extents(); }

Domain Domain::intersection(const Domain& other) const
{
  LEGATE_ASSERT(dim() == other.dim());

  auto new_lo = DomainPoint::zeroes(dim());
  auto new_hi = DomainPoint::zeroes(dim());

  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    new_lo[idx] = std::max(lo()[idx], other.lo()[idx]);
    new_hi[idx] = std::min(hi()[idx], other.hi()[idx]);
  }
  return Domain{std::move(new_lo), std::move(new_hi)};
}

std::ostream& operator<<(std::ostream& os, const Domain& domain)
{
  os << '[' << domain.lo() << ".." << domain.hi() << ']';
  return os;
}

}  // namespace legate
