/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/geometry/nd_rect.h>
#include <legate/geometry/point.h>
#include <legate/utilities/detail/doxygen.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <iterator>

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
 * @brief A dimension-erased, multi-dimensional rectangle.
 */
class LEGATE_EXPORT Rect {
 public:
  /**
   * @brief A standard-conforming forward iterator over the points of a `Rect`.
   *
   * Dereferencing the iterator yields a reference to the current `Point`.
   */
  class iterator {  // NOLINT(readability-identifier-naming)
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = Point;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const Point*;
    using reference         = const Point&;

    iterator(Point lo, Point hi, bool fortran_order, bool valid = true);

    [[nodiscard]] reference operator*() const;
    [[nodiscard]] pointer operator->() const;
    iterator& operator++();
    iterator operator++(int);
    [[nodiscard]] bool operator==(const iterator& other) const;
    [[nodiscard]] bool operator!=(const iterator& other) const;

    [[nodiscard]] const Point& lo() const;
    [[nodiscard]] const Point& hi() const;

   private:
    void step_();

    Point lo_{};
    Point hi_{};
    Point point_{};
    bool fortran_order_{false};
    bool valid_{false};
  };

  /**
   * @brief A view over the points of a `Rect`, suitable for use in range-based for loops.
   */
  class Iterable {
   public:
    Iterable(Point lo, Point hi, bool fortran_order);

    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end() const;

   private:
    Point lo_{};
    Point hi_{};
    bool fortran_order_{false};
  };

  /**
   * @brief Constructs an empty (0-dimensional) domain.
   */
  Rect() = default;

  /**
   * @brief Constructs a domain from a lower and an upper bound.
   *
   * @param lo The lower bound.
   * @param hi The upper bound. Must have the same dimensionality as `lo`.
   */
  Rect(Point lo, Point hi);

  /**
   * @brief Constructs a domain from a `legate::NDRect`
   *
   * @param rect The rectangle to convert from.
   */
  template <int DIM>
  Rect(const NDRect<DIM>& rect);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Converts this domain to a `legate::NDRect`
   *
   * The requested dimensionality must match `dim()`.
   */
  template <int DIM>
  operator NDRect<DIM>() const;  // NOLINT(google-explicit-constructor)

  /**
   * @brief Returns the bounds of this domain as a `legate::NDRect`
   *
   * The requested dimensionality must match `dim()`.
   */
  template <int DIM>
  [[nodiscard]] NDRect<DIM> bounds() const;

  [[nodiscard]] bool operator==(const Rect& rhs) const;
  [[nodiscard]] bool operator!=(const Rect& rhs) const;

  [[nodiscard]] Rect operator+(const Point& point) const;
  Rect& operator+=(const Point& point);
  [[nodiscard]] Rect operator-(const Point& point) const;
  Rect& operator-=(const Point& point);

  /**
   * @brief Checks whether the domain contains a point.
   *
   * @param point The point to test. Must have the same dimensionality as this domain.
   *
   * @return `true` if the point is within the bounds, `false` otherwise.
   */
  [[nodiscard]] bool contains(const Point& point) const;

  /**
   * @return The dimensionality of the domain.
   */
  [[nodiscard]] std::uint32_t dim() const;

  /**
   * @return `true` if the domain contains no points, `false` otherwise.
   */
  [[nodiscard]] bool empty() const;

  /**
   * @return The number of points in the domain.
   */
  [[nodiscard]] std::size_t volume() const;

  /**
   * @return A reference to the lower bound of the domain.
   */
  [[nodiscard]] const Point& lo() const;

  [[nodiscard]] Point& lo();
  /**
   * @return A reference to the upper bound of the domain.
   */
  [[nodiscard]] const Point& hi() const;

  [[nodiscard]] Point& hi();

  /**
   * @brief Intersects this domain with another.
   *
   * @param other The other domain. Must have the same dimensionality as this domain.
   *
   * @return The intersection.
   */
  [[nodiscard]] Rect intersection(const Rect& other) const;

  /**
   * @return A point whose coordinates are the extents (`hi - lo + 1`) of each dimension.
   */
  [[nodiscard]] Point extents() const;

  /**
   * @return A point whose coordinates are the C-order strides of each dimension.
   */
  [[nodiscard]] Point strides() const;

  /**
   * @return A point whose coordinates are the extents (`hi - lo + 1`) of each dimension.
   */
  [[nodiscard]] Point to_shape() const;

  // ==-- Iteration over points --==

  /**
   * @brief Returns a view over the points of the domain.
   *
   * @param fortran_order If `true`, iterate in Fortran (column-major) order; otherwise iterate in
   * C (row-major) order (the default).
   *
   * @return An iterable view, e.g. `for (auto&& point : domain.iterate()) { ... }`.
   */
  [[nodiscard]] Iterable iterate(bool fortran_order = false) const;

  [[nodiscard]] static Rect from_extents(const Point& extents);

 private:
  Point lo_{};
  Point hi_{};
};

LEGATE_EXPORT std::ostream& operator<<(std::ostream& os, const Rect& domain);

/** @} */  // end of geometry

}  // namespace legate

#include <legate/geometry/rect.inl>
