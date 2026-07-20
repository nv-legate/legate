/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/geometry/point.h>
#include <legate/utilities/detail/doxygen.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <iterator>
#include <type_traits>

/**
 * @file
 * @brief Class definition for legate::Rect
 */

namespace legate {

/**
 * @addtogroup geometry
 * @{
 */

/**
 * @brief A statically-sized, multi-dimensional rectangle.
 *
 * Each rectangle consists of two `legate::Point` objects, one for the lower bound (`lo`) and one
 * for the upper bound (`hi`). The bounds are inclusive.
 *
 * Coordinates are always stored as `std::int64_t`.
 *
 * @tparam DIM The dimensionality of the rectangle.
 */
template <int DIM>
struct Rect {
  static_assert(DIM > 0, "Rect dimensionality must be positive");

  /**
   * @brief The coordinate type.
   */
  using value_type = std::int64_t;

  /**
   * @brief A standard-conforming forward iterator over the points of a `Rect`.
   *
   * Dereferencing the iterator yields a reference to the current `legate::Point`.
   */
  class iterator {  // NOLINT(readability-identifier-naming)
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = Point<DIM>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const Point<DIM>*;
    using reference         = const Point<DIM>&;

    iterator() = default;

    LEGATE_HOST_DEVICE iterator(const Point<DIM>& lo,
                                const Point<DIM>& hi,
                                bool fortran_order,
                                bool valid);

    [[nodiscard]] LEGATE_HOST_DEVICE reference operator*() const;
    [[nodiscard]] LEGATE_HOST_DEVICE pointer operator->() const;

    LEGATE_HOST_DEVICE iterator& operator++();

    LEGATE_HOST_DEVICE iterator operator++(int);

    [[nodiscard]] LEGATE_HOST_DEVICE bool operator==(const iterator& other) const;

    [[nodiscard]] LEGATE_HOST_DEVICE bool operator!=(const iterator& other) const;

   private:
    LEGATE_HOST_DEVICE void step_();

    Point<DIM> point_{};
    Point<DIM> lo_{};
    Point<DIM> hi_{};
    bool fortran_order_{false};
    bool valid_{false};
  };

  /**
   * @brief A view over the points of a `Rect`, suitable for use in range-based for loops.
   */
  class Iterable {
   public:
    LEGATE_HOST_DEVICE Iterable(const Point<DIM>& lo, const Point<DIM>& hi, bool fortran_order);

    [[nodiscard]] LEGATE_HOST_DEVICE iterator begin() const;

    [[nodiscard]] LEGATE_HOST_DEVICE iterator end() const;

   private:
    Point<DIM> lo_{};
    Point<DIM> hi_{};
    bool fortran_order_{false};
  };

  /**
   * @brief The (inclusive) lower bound of the rectangle.
   */
  Point<DIM> lo;
  /**
   * @brief The (inclusive) upper bound of the rectangle.
   */
  Point<DIM> hi;

  /**
   * @brief Constructs an uninitialized rectangle.
   */
  Rect() = default;

  /**
   * @brief Constructs a rectangle from a lower and an upper bound.
   *
   * @param lo_ The lower bound.
   * @param hi_ The upper bound.
   */
  LEGATE_HOST_DEVICE Rect(const Point<DIM>& lo_, const Point<DIM>& hi_);

  /**
   * @return `true` if the rectangle contains no points, `false` otherwise.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE bool empty() const;

  /**
   * @return The number of points in the rectangle.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE std::size_t volume() const;

  /**
   * @brief Checks whether the rectangle contains a point.
   *
   * @param point The point to test.
   *
   * @return `true` if `point` is within the bounds, `false` otherwise.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE bool contains(const Point<DIM>& point) const;

  /**
   * @brief Checks whether the rectangle contains all points of another rectangle.
   *
   * @param other The other rectangle.
   *
   * @return `true` if every point of `other` is within this rectangle, `false` otherwise.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE bool contains(const Rect<DIM>& other) const;

  /**
   * @brief Checks whether this rectangle overlaps another.
   *
   * @param other The other rectangle.
   *
   * @return `true` if the two rectangles share at least one point, `false` otherwise.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE bool overlaps(const Rect<DIM>& other) const;

  /**
   * @brief Computes the intersection of this rectangle with another.
   *
   * @param other The other rectangle.
   *
   * @return The intersection.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE Rect<DIM> intersection(const Rect<DIM>& other) const;

  /**
   * @brief Computes the bounding box of the union of this rectangle with another.
   *
   * @param other The other rectangle.
   *
   * @return The bounding box of the union (which may itself contain points in neither rectangle).
   */
  [[nodiscard]] LEGATE_HOST_DEVICE Rect<DIM> union_bbox(const Rect<DIM>& other) const;

  // ==-- Iteration over points --==

  /**
   * @brief Returns a view over the points of the rectangle.
   *
   * @param fortran_order If `true`, iterate in Fortran (column-major) order; otherwise iterate in
   * C (row-major) order (the default).
   *
   * @return An iterable view, e.g. `for (auto&& point : rect.iterate()) { ... }`.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE Iterable iterate(bool fortran_order = false) const;
};

template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE bool operator==(const Rect<DIM>& lhs, const Rect<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE bool operator!=(const Rect<DIM>& lhs, const Rect<DIM>& rhs);

template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Rect<DIM> operator+(const Rect<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
LEGATE_HOST_DEVICE Rect<DIM>& operator+=(Rect<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
[[nodiscard]] LEGATE_HOST_DEVICE Rect<DIM> operator-(const Rect<DIM>& lhs, const Point<DIM>& rhs);
template <int DIM>
LEGATE_HOST_DEVICE Rect<DIM>& operator-=(Rect<DIM>& lhs, const Point<DIM>& rhs);

template <int DIM>
std::ostream& operator<<(std::ostream& os, const Rect<DIM>& rect);

/** @} */  // end of geometry

}  // namespace legate

#include <legate/geometry/rect.inl>
