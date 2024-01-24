/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/shared_ptr.h"
#include "core/utilities/tuple.h"
#include "core/utilities/typedefs.h"

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
 * @ingroup data
 *
 * @brief A class to express shapes of multi-dimensional entities in Legate
 *
 * Shape objects describe <em>logical</em> shapes, of multi-dimensional containers in Legate, such
 * as Legate arrays and Legate stores. For example, if the shape of a Legate store is (4, 2), the
 * store is conceptually a 2D container having four rows and two columns of elements.  The shape
 * however does not entail any particular physical manifestation of the container. The 2D store of
 * the example can be mapped to an allocation in which the elements of each row would contiguously
 * locate or an allocation in which the elements of each column would contiguously locate.
 *
 * A Shape object is essentially a tuple of extents, one for each dimension, and the dimensionality,
 * i.e., the number of dimensions, is the size of this tuple. The volume of the Shape is a product
 * of all the extents.
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
   * @brief Constructs a 0D shape
   *
   * The constructed shape is immediately ready
   *
   * Equivalent to `Shape({})`
   */
  Shape();
  /**
   * @brief Constructs the shape from a tuple of extents
   *
   * The constructed shape is immediately ready
   *
   * @param extents Dimension extents
   */
  Shape(tuple<uint64_t> extents);  // NOLINT(google-explicit-constructor)
  /**
   * @brief Constructs the shape from a vector of extents
   *
   * The constructed shape is immediately ready
   *
   * @param extents Dimension extents
   */
  explicit Shape(std::vector<uint64_t> extents);
  /**
   * @brief Constructs the shape from an initializer list of extents
   *
   * The constructed shape is immediately ready
   *
   * @param extents Dimension extents
   */
  Shape(std::initializer_list<uint64_t> extents);

  /**
   * @brief Returns the shape's extents
   *
   * If the shape is of an unbound array or store, the call blocks until the shape becomes ready.
   *
   * @return Dimension extents
   */
  [[nodiscard]] const tuple<uint64_t>& extents() const;
  /**
   * @brief Returns the shape's volume
   *
   * Equivalent to `extents().volume()`. If the shape is of an unbound array or store, the call
   * blocks until the shape becomes ready.
   *
   * @return Volume of the shape
   */
  [[nodiscard]] size_t volume() const;
  /**
   * @brief Returns the number of dimensions of this shape
   *
   * Unlike other shape-related queries, this call is non-blocking.
   *
   * @return Number of dimensions
   */
  [[nodiscard]] uint32_t ndim() const;
  /**
   * @brief Returns the extent of a given dimension
   *
   * If the shape is of an unbound array or store, the call blocks until the shape becomes ready.
   * Unlike Shape::at, this method does not check the dimension index.
   *
   * @param idx Dimension index
   *
   * @return Extent of the chosen dimension
   */
  [[nodiscard]] uint64_t operator[](uint32_t idx) const;
  /**
   * @brief Returns the extent of a given dimension
   *
   * If the shape is of an unbound array or store, the call blocks until the shape becomes ready.
   *
   * @param idx Dimension index
   *
   * @return Extent of the chosen dimension
   *
   * @throw std::out_of_range If the dimension index is invalid
   */
  [[nodiscard]] uint64_t at(uint32_t idx) const;
  /**
   * @brief Generates a human-readable string from the shape (non-blocking)
   *
   * @return String generated from the shape
   */
  [[nodiscard]] std::string to_string() const;
  /**
   * @brief Checks if this shape is the same as the given shape
   *
   * The equality check can block if one of the shapes is of an unbound array or store and the other
   * shape is not of the same container.
   *
   * @return true If the shapes are isomorphic
   * @return false Otherwise
   */
  bool operator==(const Shape& other) const;
  /**
   * @brief Checks if this shape is different from the given shape
   *
   * The equality check can block if one of the shapes is of an unbound array or store and the other
   * shape is not of the same container.
   *
   * @return true If the shapes are different
   * @return false Otherwise
   */
  bool operator!=(const Shape& other) const;

  Shape(const Shape& other)            = default;
  Shape& operator=(const Shape& other) = default;
  Shape(Shape&& other)                 = default;
  Shape& operator=(Shape&& other)      = default;

  explicit Shape(InternalSharedPtr<detail::Shape> impl);

  [[nodiscard]] const SharedPtr<detail::Shape>& impl() const;

 private:
  SharedPtr<detail::Shape> impl_{};
};

}  // namespace legate

#include "core/data/shape.inl"
