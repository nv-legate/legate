/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/geometry/domain_point.h>
#include <legate/utilities/detail/doxygen.h>

#include <realm/point.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <iterator>

/**
 * @file
 * @brief Class definition for legate::Domain
 */

namespace legate {

/**
 * @addtogroup geometry
 * @{
 */

/**
 * @brief A dimension-erased, multi-dimensional rectangle.
 */
class LEGATE_EXPORT Domain {
 public:
  /**
   * @brief The maximum number of dimensions a `Domain` may have.
   */
  static constexpr std::int32_t MAX_DIM = LEGATE_MAX_DIM;

  /**
   * @brief A standard-conforming forward iterator over the points of a `Domain`.
   *
   * Dereferencing the iterator yields a reference to the current `DomainPoint`.
   */
  class iterator {  // NOLINT(readability-identifier-naming)
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = DomainPoint;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const DomainPoint*;
    using reference         = const DomainPoint&;

    iterator(std::reference_wrapper<const DomainPoint> lo,
             std::reference_wrapper<const DomainPoint> hi,
             bool fortran_order,
             bool valid = true);

    [[nodiscard]] reference operator*() const;
    [[nodiscard]] pointer operator->() const;
    iterator& operator++();
    iterator operator++(int);
    [[nodiscard]] bool operator==(const iterator& other) const;
    [[nodiscard]] bool operator!=(const iterator& other) const;

    [[nodiscard]] const DomainPoint& lo() const;
    [[nodiscard]] const DomainPoint& hi() const;

   private:
    void step_();

    std::reference_wrapper<const DomainPoint> lo_;
    std::reference_wrapper<const DomainPoint> hi_;
    DomainPoint point_{};
    bool fortran_order_{false};
    bool valid_{false};
  };

  /**
   * @brief A view over the points of a `Domain`, suitable for use in range-based for loops.
   */
  class Iterable {
   public:
    Iterable(std::reference_wrapper<const DomainPoint> lo,
             std::reference_wrapper<const DomainPoint> hi,
             bool fortran_order);

    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end() const;

   private:
    std::reference_wrapper<const DomainPoint> lo_;
    std::reference_wrapper<const DomainPoint> hi_;
    bool fortran_order_{false};
  };

  /**
   * @brief The empty (0-dimensional) domain.
   */
  static const Domain NO_DOMAIN;

  /**
   * @brief Constructs an empty (0-dimensional) domain.
   */
  Domain() = default;

  /**
   * @brief Constructs a domain from a lower and an upper bound.
   *
   * @param lo The lower bound.
   * @param hi The upper bound. Must have the same dimensionality as `lo`.
   */
  Domain(DomainPoint lo, DomainPoint hi);

  /**
   * @brief Constructs a domain from a `legate::Rect` (a.k.a. `Realm::Rect`).
   *
   * @param rect The rectangle to convert from.
   */
  template <int DIM, typename T>
  Domain(const Realm::Rect<DIM, T>& rect);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Converts this domain to a `legate::Rect` (a.k.a. `Realm::Rect`).
   *
   * The requested dimensionality must match `dim()`.
   */
  template <int DIM, typename T>
  operator Realm::Rect<DIM, T>() const;  // NOLINT(google-explicit-constructor)

  /**
   * @brief Returns the bounds of this domain as a `legate::Rect` (a.k.a. `Realm::Rect`).
   *
   * The requested dimensionality must match `dim()`.
   */
  template <int DIM, typename T>
  [[nodiscard]] Realm::Rect<DIM, T> bounds() const;

  [[nodiscard]] bool operator==(const Domain& rhs) const;
  [[nodiscard]] bool operator!=(const Domain& rhs) const;

  [[nodiscard]] Domain operator+(const DomainPoint& point) const;
  Domain& operator+=(const DomainPoint& point);
  [[nodiscard]] Domain operator-(const DomainPoint& point) const;
  Domain& operator-=(const DomainPoint& point);

  /**
   * @brief Checks whether the domain contains a point.
   *
   * @param point The point to test. Must have the same dimensionality as this domain.
   *
   * @return `true` if the point is within the bounds, `false` otherwise.
   */
  [[nodiscard]] bool contains(const DomainPoint& point) const;

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
  [[nodiscard]] const DomainPoint& lo() const;

  /**
   * @return A reference to the upper bound of the domain.
   */
  [[nodiscard]] const DomainPoint& hi() const;

  /**
   * @brief Intersects this domain with another.
   *
   * @param other The other domain. Must have the same dimensionality as this domain.
   *
   * @return The intersection.
   */
  [[nodiscard]] Domain intersection(const Domain& other) const;

  /**
   * @return A point whose coordinates are the extents (`hi - lo + 1`) of each dimension.
   */
  [[nodiscard]] DomainPoint extents() const;

  /**
   * @return A point whose coordinates are the C-order strides of each dimension.
   */
  [[nodiscard]] DomainPoint strides() const;

  /**
   * @return A point whose coordinates are the extents (`hi - lo + 1`) of each dimension.
   */
  [[nodiscard]] DomainPoint to_shape() const;

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

 private:
  DomainPoint lo_{};
  DomainPoint hi_{};
};

LEGATE_EXPORT std::ostream& operator<<(std::ostream& os, const Domain& domain);

/** @} */  // end of geometry

}  // namespace legate

#include <legate/geometry/domain.inl>
