/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/geometry/domain_point.h>

#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/zip.h>

#include <cstddef>
#include <ostream>
#include <utility>

namespace legate {

DomainPoint::DomainPoint(detail::SmallVector<std::int64_t, LEGATE_MAX_DIM> coords)
  : coordinates_{std::move(coords)}
{
}

DomainPoint::DomainPoint(Span<const std::int64_t> coords) : coordinates_{coords} {}

/*static*/ DomainPoint DomainPoint::zeroes(std::int32_t ndim)
{
  return DomainPoint{detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{
    detail::tags::size_tag, static_cast<std::size_t>(ndim), std::int64_t{0}}};
}

std::int64_t DomainPoint::dot(const DomainPoint& rhs) const
{
  std::int64_t result = 0;

  for (auto&& [l, r] : detail::zip_equal(*this, rhs)) {
    result += l * r;
  }
  return result;
}

std::int64_t DomainPoint::volume() const
{
  std::int64_t result = 1;

  for (auto&& coord : *this) {
    result *= coord;
  }
  return result;
}

DomainPoint DomainPoint::strides() const
{
  auto result = DomainPoint::zeroes(dim());

  if (dim() == 0) {
    return result;
  }

  // TODO(wonchanl): C order is assumed but the layout can be arbitrary
  result[dim() - 1] = 1;
  for (std::int32_t idx = dim() - 2; idx >= 0; --idx) {
    const auto next = idx + 1;

    result[idx] = result[next] * coordinates_[next];
  }
  return result;
}

DomainPoint DomainPoint::inserted(std::int32_t pos, std::int64_t value) const
{
  const auto new_dim = static_cast<std::int32_t>(dim()) + 1;
  auto result        = DomainPoint::zeroes(new_dim);

  for (std::int32_t i = 0, j = 0; i < new_dim; ++i) {
    result[i] = (i == pos) ? value : coordinates_[j++];
  }
  return result;
}

DomainPoint DomainPoint::removed(std::int32_t pos) const
{
  const auto old_dim = static_cast<std::int32_t>(dim());
  auto result        = DomainPoint::zeroes(old_dim - 1);

  for (std::int32_t i = 0, j = 0; i < old_dim; ++i) {
    if (i != pos) {
      result[j++] = coordinates_[i];
    }
  }
  return result;
}

DomainPoint DomainPoint::permuted(Span<const std::int32_t> axes) const
{
  auto result = DomainPoint::zeroes(static_cast<std::int32_t>(axes.size()));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(axes.size()); ++i) {
    result[i] = coordinates_[static_cast<std::size_t>(axes[static_cast<std::size_t>(i)])];
  }
  return result;
}

std::ostream& operator<<(std::ostream& os, const DomainPoint& point)
{
  os << '(';
  for (std::uint32_t idx = 0; idx < point.dim(); ++idx) {
    if (idx > 0) {
      os << ',';
    }
    os << point[idx];
  }
  os << ')';
  return os;
}

bool DomainPoint::operator==(const DomainPoint& rhs) const
{
  if (dim() != rhs.dim()) {
    return false;
  }
  for (auto&& [l, r] : detail::zip_equal(*this, rhs)) {
    if (l != r) {
      return false;
    }
  }
  return true;
}

bool DomainPoint::operator!=(const DomainPoint& rhs) const { return !(*this == rhs); }

DomainPoint DomainPoint::operator+(std::int64_t scalar) const
{
  auto result = *this;

  result += scalar;
  return result;
}

DomainPoint DomainPoint::operator+(const DomainPoint& rhs) const
{
  auto result = *this;

  result += rhs;
  return result;
}

DomainPoint& DomainPoint::operator+=(std::int64_t scalar)
{
  for (auto& coord : coordinates_) {
    coord += scalar;
  }
  return *this;
}

DomainPoint& DomainPoint::operator+=(const DomainPoint& rhs)
{
  LEGATE_ASSERT(dim() == rhs.dim());
  for (auto&& [dim, rh] : detail::enumerate(rhs)) {
    coordinates_[dim] += rh;
  }
  return *this;
}

DomainPoint DomainPoint::operator-(std::int64_t scalar) const
{
  auto result = *this;

  result -= scalar;
  return result;
}

DomainPoint DomainPoint::operator-(const DomainPoint& rhs) const
{
  auto result = *this;

  result -= rhs;
  return result;
}

DomainPoint& DomainPoint::operator-=(std::int64_t scalar)
{
  for (auto& coord : coordinates_) {
    coord -= scalar;
  }
  return *this;
}

DomainPoint& DomainPoint::operator-=(const DomainPoint& rhs)
{
  LEGATE_ASSERT(dim() == rhs.dim());
  for (auto&& [dim, rh] : detail::enumerate(rhs)) {
    coordinates_[dim] -= rh;
  }
  return *this;
}

DomainPoint DomainPoint::operator*(std::int64_t scalar) const
{
  auto result = *this;

  result *= scalar;
  return result;
}

DomainPoint DomainPoint::operator*(const DomainPoint& rhs) const
{
  auto result = *this;

  result *= rhs;
  return result;
}

DomainPoint& DomainPoint::operator*=(std::int64_t scalar)
{
  for (auto& coord : coordinates_) {
    coord *= scalar;
  }
  return *this;
}

DomainPoint& DomainPoint::operator*=(const DomainPoint& rhs)
{
  LEGATE_ASSERT(dim() == rhs.dim());
  for (auto&& [dim, rh] : detail::enumerate(rhs)) {
    coordinates_[dim] *= rh;
  }
  return *this;
}

DomainPoint DomainPoint::operator/(std::int64_t scalar) const
{
  auto result = *this;

  result /= scalar;
  return result;
}

DomainPoint DomainPoint::operator/(const DomainPoint& rhs) const
{
  auto result = *this;

  result /= rhs;
  return result;
}

DomainPoint& DomainPoint::operator/=(std::int64_t scalar)
{
  for (auto& coord : coordinates_) {
    coord /= scalar;
  }
  return *this;
}

DomainPoint& DomainPoint::operator/=(const DomainPoint& rhs)
{
  LEGATE_ASSERT(dim() == rhs.dim());
  for (auto&& [dim, rh] : detail::enumerate(rhs)) {
    coordinates_[dim] /= rh;
  }
  return *this;
}

DomainPoint DomainPoint::operator%(std::int64_t scalar) const
{
  auto result = *this;

  result %= scalar;
  return result;
}

DomainPoint DomainPoint::operator%(const DomainPoint& rhs) const
{
  auto result = *this;

  result %= rhs;
  return result;
}

DomainPoint& DomainPoint::operator%=(std::int64_t scalar)
{
  for (auto& coord : coordinates_) {
    coord %= scalar;
  }
  return *this;
}

DomainPoint& DomainPoint::operator%=(const DomainPoint& rhs)
{
  LEGATE_ASSERT(dim() == rhs.dim());
  for (auto&& [dim, rh] : detail::enumerate(rhs)) {
    coordinates_[dim] %= rh;
  }
  return *this;
}

}  // namespace legate
