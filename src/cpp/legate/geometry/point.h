/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/geometry/nd_point.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>

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
 * @brief A dimension-erased, multi-dimensional point.
 */
class LEGATE_EXPORT Point {
 public:
  /**
   * @brief The maximum number of dimensions a `Point` may have.
   */
  static constexpr std::int32_t MAX_DIM = LEGATE_MAX_DIM;

  /**
   * @brief Constructs an empty (0-dimensional) point.
   */
  Point() = default;
  /**
   * @brief Constructs a 1-dimensional point holding a single coordinate.
   *
   * @param index The coordinate value.
   */
  Point(std::int64_t index);  // NOLINT(google-explicit-constructor)
  /**
   * @brief Constructs a point from a vector of coordinates.
   *
   * @param coords The coordinates. The dimensionality of the point is `coords.size()`.
   */
  explicit Point(detail::SmallVector<std::int64_t, LEGATE_MAX_DIM> coords);
  /**
   * @brief Constructs a point from a vector of coordinates.
   *
   * @param coords The coordinates. The dimensionality of the point is `coords.size()`.
   */
  explicit Point(const detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>& extents);
  /**
   * @brief Constructs a point from a span of coordinates.
   *
   * @param coords The coordinates. The dimensionality of the point is `coords.size()`.
   */
  explicit Point(Span<const std::int64_t> coords);
  /**
   * @brief Constructs a point from a `legate::NDPoint`.
   *
   * @param point The point to convert from.
   */
  template <int DIM>
  Point(const NDPoint<DIM>& point);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Converts this point to a `legate::NDPoint`.
   *
   * The requested dimensionality must match `dim()`.
   */
  template <int DIM>
  operator NDPoint<DIM>() const;  // NOLINT(google-explicit-constructor)

  template <int DIM>
  Point& operator=(const NDPoint<DIM>& point);

  /**
   * @return The dimensionality of the point.
   */
  [[nodiscard]] std::uint32_t dim() const;

  /**
   * @return The number of coordinates in the point (equal to `dim()`).
   *
   * @note Provided so a `Point` can stand in for a container of coordinates in generic code.
   */
  [[nodiscard]] std::size_t size() const;

  /**
   * @return `true` if the point is 0-dimensional (has no coordinates), `false` otherwise.
   */
  [[nodiscard]] bool empty() const;

  // ==-- Utility methods (formerly free functions in geometry.h) --==

  /**
   * @brief Computes the inner (dot) product of this point with another.
   *
   * @param other The other point. Must have the same dimensionality as this point.
   *
   * @return The inner product.
   */
  [[nodiscard]] std::int64_t dot(const Point& other) const;

  /**
   * @brief Computes the volume of a shape described by this point.
   *
   * @return The product of all coordinates.
   */
  [[nodiscard]] std::int64_t volume() const;

  /**
   * @brief Computes the (C-order) strides for a shape described by this point.
   *
   * @return A point whose coordinates are the strides for each dimension.
   */
  [[nodiscard]] Point strides() const;

  [[nodiscard]] bool operator==(const Point& rhs) const;
  [[nodiscard]] bool operator!=(const Point& rhs) const;

  [[nodiscard]] Point operator+(std::int64_t scalar) const;
  [[nodiscard]] Point operator+(const Point& rhs) const;
  Point& operator+=(std::int64_t scalar);
  Point& operator+=(const Point& rhs);

  [[nodiscard]] Point operator-(std::int64_t scalar) const;
  [[nodiscard]] Point operator-(const Point& rhs) const;
  Point& operator-=(std::int64_t scalar);
  Point& operator-=(const Point& rhs);

  [[nodiscard]] Point operator*(std::int64_t scalar) const;
  [[nodiscard]] Point operator*(const Point& rhs) const;
  Point& operator*=(std::int64_t scalar);
  Point& operator*=(const Point& rhs);

  [[nodiscard]] Point operator/(std::int64_t scalar) const;
  [[nodiscard]] Point operator/(const Point& rhs) const;
  Point& operator/=(std::int64_t scalar);
  Point& operator/=(const Point& rhs);

  [[nodiscard]] Point operator%(std::int64_t scalar) const;
  [[nodiscard]] Point operator%(const Point& rhs) const;
  Point& operator%=(std::int64_t scalar);
  Point& operator%=(const Point& rhs);

  template <typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
  [[nodiscard]] std::int64_t& operator[](Index index);
  template <typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
  [[nodiscard]] const std::int64_t& operator[](Index index) const;

  /**
   * @brief Creates a zero-filled point of the given dimensionality.
   *
   * @param ndim The dimensionality of the point.
   *
   * @return The newly created point.
   */
  [[nodiscard]] static Point zeroes(std::int32_t ndim);
  /**
   * @brief Creates a zero-filled point of the given dimensionality.
   *
   * @param ndim The dimensionality of the point.
   *
   * @return The newly created point.
   */
  [[nodiscard]] static Point full(std::int32_t ndim, std::int64_t coord);

  /**
   * @brief Returns a copy of this point with `value` inserted at dimension `pos`.
   *
   * The result has one more dimension than this point.
   *
   * @param pos The dimension at which to insert (`0 <= pos <= dim()`).
   * @param value The coordinate value to insert.
   *
   * @return The higher-dimensional point.
   */
  [[nodiscard]] Point inserted(std::int32_t pos, std::int64_t value) const;

  /**
   * @brief Returns a copy of this point with dimension `pos` removed.
   *
   * The result has one fewer dimension than this point.
   *
   * @param pos The dimension to remove (`0 <= pos < dim()`).
   *
   * @return The lower-dimensional point.
   */
  [[nodiscard]] Point removed(std::int32_t pos) const;

  /**
   * @brief Returns a copy of this point with its coordinates gathered by `axes`.
   *
   * The result has `axes.size()` dimensions, with `result[i] == (*this)[axes[i]]`.
   *
   * @param axes The source dimension for each result dimension.
   *
   * @return The permuted point.
   */
  [[nodiscard]] Point permuted(Span<const std::int32_t> axes) const;

  // ==-- Iteration over coordinates --==

  [[nodiscard]] const std::int64_t* begin() const;
  [[nodiscard]] const std::int64_t* end() const;

 private:
  detail::SmallVector<std::int64_t, LEGATE_MAX_DIM> coordinates_{};
};

LEGATE_EXPORT std::ostream& operator<<(std::ostream& os, const Point& point);

/** @} */  // end of geometry

}  // namespace legate

#include <legate/geometry/point.inl>
