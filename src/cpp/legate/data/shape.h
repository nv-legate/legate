/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/tuple.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

/**
 * @file
 * @brief Class definition for legate::Shape
 */

namespace legate {

namespace detail {

class Shape;

}  // namespace detail

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief A class to express shapes of multi-dimensional entities in Legate
 *
 * `Shape` objects describe *logical* shapes, of multi-dimensional containers in Legate such as
 * Legate arrays and Legate stores. For example, if the shape of a Legate store is `(4, 2)`,
 * the store is conceptually a 2D container having four rows and two columns of elements.  The
 * shape however does not entail any particular physical manifestation of the container. The
 * aforementioned 2D store can be mapped to an allocation in which the elements of each row
 * would be contiguously located or an allocation in which the elements of each column would
 * be contiguously located.
 *
 * A `Shape` object is essentially a tuple of extents, one for each dimension, and the
 * dimensionality, i.e., the number of dimensions, is the size of this tuple. The volume of the
 * `Shape` is a product of all the extents.
 *
 * Since Legate allows containers' shapes to be determined by tasks, some shapes may not be "ready"
 * when the control code tries to introspect their extents. In this case, the control code will be
 * blocked until the tasks updating the containers are complete. This asynchrony behind the shape
 * objects is hidden from the control code and it's recommended that introspection of the shapes of
 * unbound arrays or stores should be avoided. The blocking behavior of each API call can be found
 * in its reference (methods with no mention of blocking should exhibit no shape-related blocking).
 */
class Shape {
 public:
  /**
   * @brief Constructs a 0D `Shape`
   *
   * The constructed `Shape` is immediately ready
   *
   * Equivalent to `Shape({})`
   */
  Shape();

  /**
   * @brief Constructs a `Shape` from a `tuple` of extents
   *
   * The constructed `Shape` is immediately ready
   *
   * @param extents Dimension extents
   */
  Shape(tuple<std::uint64_t> extents);  // NOLINT(google-explicit-constructor)

  /**
   * @brief Constructs a `Shape` from a `std::vector` of extents
   *
   * The constructed `Shape` is immediately ready
   *
   * @param extents Dimension extents
   */
  explicit Shape(std::vector<std::uint64_t> extents);

  /**
   * @brief Constructs a `Shape` from a `std::initializer_list` of extents
   *
   * The constructed `Shape` is immediately ready
   *
   * @param extents Dimension extents
   */
  Shape(std::initializer_list<std::uint64_t> extents);

  /**
   * @brief Returns the `Shape`'s extents
   *
   * If the `Shape` is of an unbound array or store, the call blocks until the shape becomes ready.
   *
   * @return Dimension extents
   */
  [[nodiscard]] const tuple<std::uint64_t>& extents() const;

  /**
   * @brief Returns the `Shape`'s volume
   *
   * Equivalent to `extents().volume()`. If the `Shape` is of an unbound array or store, the call
   * blocks until the `Shape` becomes ready.
   *
   * @return Volume of the `Shape`
   */
  [[nodiscard]] std::size_t volume() const;

  /**
   * @brief Returns the number of dimensions of this `Shape`
   *
   * Unlike other `Shape`-related queries, this call is non-blocking.
   *
   * @return Number of dimensions
   */
  [[nodiscard]] std::uint32_t ndim() const;

  /**
   * @brief Returns the extent of a given dimension
   *
   * If the `Shape` is of an unbound array or store, the call blocks until the `Shape` becomes
   * ready. Unlike `Shape::at()`, this method does not check the dimension index.
   *
   * @param idx Dimension index
   *
   * @return Extent of the chosen dimension
   */
  [[nodiscard]] std::uint64_t operator[](std::uint32_t idx) const;

  /**
   * @brief Returns the extent of a given dimension
   *
   * If the `Shape` is of an unbound array or store, the call blocks until the `Shape` becomes
   * ready.
   *
   * @param idx Dimension index
   *
   * @return Extent of the chosen dimension
   *
   * @throw std::out_of_range If the dimension index is invalid
   */
  [[nodiscard]] std::uint64_t at(std::uint32_t idx) const;

  /**
   * @brief Generates a human-readable string from the `Shape` (non-blocking)
   *
   * @return `std::string` generated from the `Shape`
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @brief Checks if this `Shape` is the same as the given `Shape`
   *
   * The equality check can block if one of the `Shape`s is of an unbound array or store and the
   * other `Shape` is not of the same container.
   *
   * @return `true` if the `Shape`s are isomorphic, `false` otherwise
   */
  [[nodiscard]] bool operator==(const Shape& other) const;

  /**
   * @brief Checks if this `Shape` is different from the given `Shape`
   *
   * The equality check can block if one of the `Shape`s is of an unbound array or store and
   * the other `Shape` is not of the same container.
   *
   * @return `true` if the `Shape`s are different, `false` otherwise
   */
  [[nodiscard]] bool operator!=(const Shape& other) const;

  Shape(const Shape& other)            = default;
  Shape& operator=(const Shape& other) = default;
  Shape(Shape&& other)                 = default;
  Shape& operator=(Shape&& other)      = default;

  explicit Shape(InternalSharedPtr<detail::Shape> impl);

  [[nodiscard]] const SharedPtr<detail::Shape>& impl() const;

 private:
  SharedPtr<detail::Shape> impl_{};
};

/** @} */

}  // namespace legate

#include <legate/data/shape.inl>
