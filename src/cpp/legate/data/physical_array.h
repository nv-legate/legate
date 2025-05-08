/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/logical_array.h>
#include <legate/data/physical_store.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>

/**
 * @file
 * @brief Class definition for legate::PhysicalArray
 */

namespace legate {

namespace detail {

class PhysicalArray;

}  // namespace detail

class ListPhysicalArray;
class StringPhysicalArray;

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief A multi-dimensional array abstraction for fixed- or variable-size elements
 *
 * `PhysicalArray`s can be backed by one or more `PhysicalStore`s, depending on their types.
 */
class PhysicalArray {
 public:
  /**
   * @brief Indicates if the array is nullable
   *
   * @return `true` if the array is nullable, `false` otherwise
   */
  [[nodiscard]] bool nullable() const noexcept;

  /**
   * @brief Returns the dimension of the array
   *
   * @return Array's dimension
   */
  [[nodiscard]] std::int32_t dim() const noexcept;

  /**
   * @brief Returns the array's `Type`
   *
   * @return `Type`
   */
  [[nodiscard]] Type type() const noexcept;

  /**
   * @brief Indicates if the array has child arrays
   *
   * @return `true` if the array has child arrays, `false` otherwise
   */
  [[nodiscard]] bool nested() const noexcept;

  /**
   * @brief Returns the store containing the array's data
   *
   * @return `PhysicalStore`
   *
   * @throw std::invalid_argument If the array is not a base array
   */
  [[nodiscard]] PhysicalStore data() const;

  /**
   * @brief Returns the store containing the array's null mask
   *
   * @return `PhysicalStore`
   *
   * @throw std::invalid_argument If the array is not nullable
   */
  [[nodiscard]] PhysicalStore null_mask() const;

  /**
   * @brief Returns the sub-array of a given index
   *
   * @param index Sub-array index
   *
   * @return Sub-`PhysicalArray` at the given index
   *
   * @throw std::invalid_argument If the array has no child arrays
   * @throw std::out_of_range If the index is out of range
   */
  [[nodiscard]] PhysicalArray child(std::uint32_t index) const;

  /**
   * @brief Returns the array's shape
   *
   * @return Array's shape
   */
  template <std::int32_t DIM>
  [[nodiscard]] Rect<DIM> shape() const;

  /**
   * @brief Returns the array's `Domain`
   *
   * @return Array's `Domain`
   */
  [[nodiscard]] Domain domain() const;

  /**
   * @brief Casts this array as a `ListPhysicalArray`
   *
   * @return This array as a `ListPhysicalArray`
   *
   * @throw std::invalid_argument If the array is not a list array
   */
  [[nodiscard]] ListPhysicalArray as_list_array() const;

  /**
   * @brief Casts this array as a `StringPhysicalArray`
   *
   * @return This array as a `StringPhysicalArray`
   *
   * @throw std::invalid_argument If the array is not a string array
   */
  [[nodiscard]] StringPhysicalArray as_string_array() const;

  explicit PhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl,
                         std::optional<LogicalArray> owner = std::nullopt);

  [[nodiscard]] const SharedPtr<detail::PhysicalArray>& impl() const;

  PhysicalArray() = LEGATE_DEFAULT_WHEN_CYTHON;

  virtual ~PhysicalArray() noexcept                  = default;
  PhysicalArray(const PhysicalArray&) noexcept       = default;
  PhysicalArray& operator=(const PhysicalArray&)     = default;
  PhysicalArray(PhysicalArray&&) noexcept            = default;
  PhysicalArray& operator=(PhysicalArray&&) noexcept = default;

  [[nodiscard]] const std::optional<LogicalArray>& owner() const;

 private:
  void check_shape_dimension_(std::int32_t dim) const;

 protected:
  SharedPtr<detail::PhysicalArray> impl_{};

 private:
  // This member exists purely to solve the temporary array problem. It is illegal for Physical
  // arrays to outlive their LogicalArray counterparts, but it is pretty easy to get into a
  // situation where this happens. For example, you could do:
  //
  // auto phys = get_runtime()->create_array(...).get_physical_array();
  //
  // While this is illegal from the runtime perspective, we still want to make this "work" from
  // a user perspective, as it is very easy to get into. So we have this member. It's value is
  // immaterial (and should not be relied upon), and isn't exposed anywhere else.
  std::optional<LogicalArray> owner_{};
};

/**
 * @brief A multi-dimensional array abstraction for variable-size list of elements.
 */
class ListPhysicalArray : public PhysicalArray {
 public:
  /**
   * @brief Returns the sub-array for descriptors
   *
   * @return `PhysicalArray` of descriptors
   */
  [[nodiscard]] PhysicalArray descriptor() const;

  /**
   * @brief Returns the sub-array for variable size data
   *
   * @return `PhysicalArray` of variable sized data
   */
  [[nodiscard]] PhysicalArray vardata() const;

 private:
  friend class PhysicalArray;

  explicit ListPhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl,
                             std::optional<LogicalArray> owner);
};

/**
 * @brief A multi-dimensional array abstraction representing a string.
 */
class StringPhysicalArray : public PhysicalArray {
 public:
  /**
   * @brief Returns the sub-array for ranges
   *
   * @return `PhysicalArray` of ranges
   */
  [[nodiscard]] PhysicalArray ranges() const;

  /**
   * @brief Returns the sub-array for characters
   *
   * @return `PhysicalArray` of the characters in the string.
   */
  [[nodiscard]] PhysicalArray chars() const;

 private:
  friend class PhysicalArray;

  explicit StringPhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl,
                               std::optional<LogicalArray> owner);
};

/** @} */

}  // namespace legate

#include <legate/data/physical_array.inl>
