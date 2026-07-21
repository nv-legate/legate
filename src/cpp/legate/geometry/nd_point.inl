/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/nd_point.h>

#include <ostream>

namespace legate {

template <int DIM>
template <int D, std::enable_if_t<D == 1, int>>
LEGATE_HOST_DEVICE NDPoint<DIM>::NDPoint(value_type val)  // NOLINT(google-explicit-constructor)
{
  values[0] = val;
}

template <int DIM>
template <int D, std::enable_if_t<(D > 1), int>>
LEGATE_HOST_DEVICE NDPoint<DIM>::NDPoint(value_type val)
{
  for (int idx = 0; idx < DIM; ++idx) {
    values[idx] = val;
  }
}

template <int DIM>
template <typename Arg0,
          typename Arg1,
          typename... Args,
          std::enable_if_t<(sizeof...(Args) + 2 <= DIM), int>>
LEGATE_HOST_DEVICE NDPoint<DIM>::NDPoint(Arg0 val0, Arg1 val1, Args... vals)
  : values{static_cast<value_type>(val0),
           static_cast<value_type>(val1),
           static_cast<value_type>(vals)...}
{
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM>::NDPoint(const value_type* vals)
{
  for (int idx = 0; idx < DIM; ++idx) {
    values[idx] = vals[idx];
  }
}

template <int DIM>
template <int D, std::enable_if_t<D == 1, int>>
LEGATE_HOST_DEVICE NDPoint<DIM>::operator value_type() const  // NOLINT(google-explicit-constructor)
{
  return values[0];
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDPoint<DIM>::value_type& NDPoint<DIM>::operator[](int index)
{
  return values[index];
}

template <int DIM>
LEGATE_HOST_DEVICE const typename NDPoint<DIM>::value_type& NDPoint<DIM>::operator[](
  int index) const
{
  return values[index];
}

template <int DIM>
LEGATE_HOST_DEVICE typename NDPoint<DIM>::value_type NDPoint<DIM>::dot(
  const NDPoint<DIM>& rhs) const
{
  value_type acc = values[0] * rhs[0];

  for (int idx = 1; idx < DIM; ++idx) {
    acc += values[idx] * rhs[idx];
  }
  return acc;
}

template <int DIM>
LEGATE_HOST_DEVICE constexpr NDPoint<DIM> NDPoint<DIM>::zeroes()
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result.values[idx] = 0;
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE constexpr NDPoint<DIM> NDPoint<DIM>::ones()
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result.values[idx] = 1;
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE bool operator==(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  for (int idx = 0; idx < DIM; ++idx) {
    if (lhs[idx] != rhs[idx]) {
      return false;
    }
  }
  return true;
}

template <int DIM>
LEGATE_HOST_DEVICE bool operator!=(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  return !(lhs == rhs);
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM> operator+(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result[idx] = lhs[idx] + rhs[idx];
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM>& operator+=(NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  for (int idx = 0; idx < DIM; ++idx) {
    lhs[idx] += rhs[idx];
  }
  return lhs;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM> operator-(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result[idx] = lhs[idx] - rhs[idx];
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM>& operator-=(NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  for (int idx = 0; idx < DIM; ++idx) {
    lhs[idx] -= rhs[idx];
  }
  return lhs;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM> operator*(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result[idx] = lhs[idx] * rhs[idx];
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM>& operator*=(NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  for (int idx = 0; idx < DIM; ++idx) {
    lhs[idx] *= rhs[idx];
  }
  return lhs;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM> operator/(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result[idx] = lhs[idx] / rhs[idx];
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM> operator/(const NDPoint<DIM>& lhs,
                                          typename NDPoint<DIM>::value_type rhs)
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result[idx] = lhs[idx] / rhs;
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM>& operator/=(NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  for (int idx = 0; idx < DIM; ++idx) {
    lhs[idx] /= rhs[idx];
  }
  return lhs;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM> operator%(const NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  NDPoint<DIM> result{};

  for (int idx = 0; idx < DIM; ++idx) {
    result[idx] = lhs[idx] % rhs[idx];
  }
  return result;
}

template <int DIM>
LEGATE_HOST_DEVICE NDPoint<DIM>& operator%=(NDPoint<DIM>& lhs, const NDPoint<DIM>& rhs)
{
  for (int idx = 0; idx < DIM; ++idx) {
    lhs[idx] %= rhs[idx];
  }
  return lhs;
}

template <int DIM>
std::ostream& operator<<(std::ostream& os, const NDPoint<DIM>& point)
{
  os << '<' << point[0];
  for (int idx = 1; idx < DIM; ++idx) {
    os << ',' << point[idx];
  }
  os << '>';
  return os;
}

}  // namespace legate
