/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <type_traits>

/**
 * @file
 * @brief Class definition for legate::Point
 */

namespace legate {

/**
 * @addtogroup geometry
 * @{
 */

/**
 * @brief A statically-sized, multi-dimensional point.
 *
 * `Point` objects support index expressions; they can be accessed like a statically-sized array.
 * `Point` objects also support the usual component-wise arithmetic operators and a `dot` operator.
 *
 * Coordinates are always stored as `std::int64_t`.
 *
 * @tparam DIM The dimensionality of the point.
 */
template <int DIM>
struct Point {
  /**
   * @brief The coordinate type.
   */
  using value_type = std::int64_t;

  /**
   * @brief The coordinates of the point. Prefer `operator[]` over accessing this directly.
   */
  value_type values[DIM];  // NOLINT(*-avoid-c-arrays)

  /**
   * @brief Constructs an uninitialized point.
   */
  Point() = default;

  /**
   * @brief Constructs a point whose coordinates are all `val`.
   *
   * This constructor is implicit for `DIM == 1` (so that a coordinate may be used wherever a
   * 1-dimensional point is expected, e.g. `Rect<1>{0, n}`) and explicit otherwise.
   *
   * @param val The value to broadcast across all coordinates.
   */
  template <int D = DIM, std::enable_if_t<D == 1, int> = 0>
  LEGATE_HOST_DEVICE Point(value_type val);  // NOLINT(google-explicit-constructor)

  template <int D = DIM, std::enable_if_t<(D > 1), int> = 0>
  LEGATE_HOST_DEVICE explicit Point(value_type val);

  /**
   * @brief Constructs a point from a list of coordinates.
   *
   * @param val0 The first coordinate.
   * @param val1 The second coordinate.
   * @param vals The remaining coordinates.
   */
  template <typename Arg0,
            typename Arg1,
            typename... Args,
            std::enable_if_t<(sizeof...(Args) + 2 <= DIM), int> = 0>
  LEGATE_HOST_DEVICE Point(Arg0 val0, Arg1 val1, Args... vals);

  /**
   * @brief Constructs a point from an array of coordinates.
   *
   * @param vals Pointer to at least `DIM` coordinates.
   */
  LEGATE_HOST_DEVICE explicit Point(const value_type* vals);

  /**
   * @brief Converts a 1-dimensional point to its sole coordinate.
   *
   * This conversion is available only for `DIM == 1`, so that a 1-dimensional point may be used
   * wherever a coordinate is expected.
   */
  template <int D = DIM, std::enable_if_t<D == 1, int> = 0>
  // NOLINTNEXTLINE(google-explicit-constructor)
  [[nodiscard]] LEGATE_HOST_DEVICE operator value_type() const;

  [[nodiscard]] LEGATE_HOST_DEVICE value_type& operator[](int index);
  [[nodiscard]] LEGATE_HOST_DEVICE const value_type& operator[](int index) const;

  /**
   * @brief Computes the inner (dot) product of this point with another.
   *
   * @param rhs The other point.
   *
   * @return The inner product.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE value_type dot(const Point<DIM>& rhs) const;

  /**
   * @brief Creates a point whose coordinates are all zero.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE static constexpr Point<DIM> zeroes();

  /**
   * @brief Creates a point whose coordinates are all one.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE static constexpr Point<DIM> ones();
};

template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE bool operator==(const Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE bool operator!=(const Point<DIM>& lhs, const Point<DIM>& rhs);

template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Point<DIM> operator+(const Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
LEGATE_HOST_DEVICE Point<DIM>& operator+=(Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Point<DIM> operator-(const Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
LEGATE_HOST_DEVICE Point<DIM>& operator-=(Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Point<DIM> operator*(const Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
LEGATE_HOST_DEVICE Point<DIM>& operator*=(Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Point<DIM> operator/(const Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Point<DIM> operator/(const Point<DIM>& lhs,
                                                      typename Point<DIM>::value_type rhs);
template <int DIM>
LEGATE_HOST_DEVICE Point<DIM>& operator/=(Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Point<DIM> operator%(const Point<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
LEGATE_HOST_DEVICE Point<DIM>& operator%=(Point<DIM>& lhs, const Point<DIM>& rhs);

template <int DIM>
std::ostream& operator<<(std::ostream& os, const Point<DIM>& point);

/** @} */  // end of geometry

}  // namespace legate

#include <legate/geometry/point.inl>
