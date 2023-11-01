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

#include "core/data/store.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

#include <cstdint>
#include <memory>

/**
 * @file
 * @brief Class definition for legate::Array
 */

namespace legate {

namespace detail {
struct Array;
}  // namespace detail

class ListArray;
class StringArray;

class Array {
 public:
  /**
   * @brief Indicates if the array is nullable
   *
   * @return true If the array is nullable
   * @return false Otherwise
   */
  [[nodiscard]] bool nullable() const noexcept;
  /**
   * @brief Returns the dimension of the array
   *
   * @return Array's dimension
   */
  [[nodiscard]] int32_t dim() const noexcept;
  /**
   * @brief Returns the array's type
   *
   * @return Type
   */
  [[nodiscard]] Type type() const /* noexcept? */;
  /**
   * @brief Indicates if the array has child arrays
   *
   * @return true If the array has child arrays
   * @return false Otherwise
   */
  [[nodiscard]] bool nested() const noexcept;

  /**
   * @brief Returns the store containing the array's data
   *
   * @return Store
   *
   * @throw std::invalid_argument If the array is not a base array
   */
  [[nodiscard]] Store data() const;
  /**
   * @brief Returns the store containing the array's null mask
   *
   * @return Store
   *
   * @throw std::invalid_argument If the array is not nullable
   */
  [[nodiscard]] Store null_mask() const;
  /**
   * @brief Returns the sub-array of a given index
   *
   * @param index Sub-array index
   *
   * @return Array
   *
   * @throw std::invalid_argument If the array has no child arrays
   * @throw std::out_of_range If the index is out of range
   */
  [[nodiscard]] Array child(uint32_t index) const;

  /**
   * @brief Returns the array's domain
   *
   * @return Array's domain
   */
  template <int32_t DIM>
  [[nodiscard]] Rect<DIM> shape() const;
  /**
   * @brief Returns the array's domain in a dimension-erased domain type
   *
   * @return Array's domain in a dimension-erased domain type
   */
  [[nodiscard]] Domain domain() const;

  /**
   * @brief Casts this array as a list array
   *
   * @return List array
   *
   * @throw std::invalid_argument If the array is not a list array
   */
  [[nodiscard]] ListArray as_list_array() const;
  /**
   * @brief Casts this array as a string array
   *
   * @return String array
   *
   * @throw std::invalid_argument If the array is not a string array
   */
  [[nodiscard]] StringArray as_string_array() const;

  explicit Array(std::shared_ptr<detail::Array> impl);

  [[nodiscard]] const std::shared_ptr<detail::Array>& impl() const;

  Array(const Array&) noexcept = default;
  Array(Array&&) noexcept      = default;

  virtual ~Array() noexcept = default;

 private:
  void check_shape_dimension(int32_t dim) const;

 protected:
  std::shared_ptr<detail::Array> impl_{};
};

class ListArray : public Array {
 public:
  /**
   * @brief Returns the sub-array for descriptors
   *
   * @return Store
   */
  [[nodiscard]] Array descriptor() const;
  /**
   * @brief Returns the sub-array for variable size data
   *
   * @return Store
   */
  [[nodiscard]] Array vardata() const;

 private:
  friend class Array;

  explicit ListArray(std::shared_ptr<detail::Array> impl);
};

class StringArray : public Array {
 public:
  /**
   * @brief Returns the sub-array for ranges
   *
   * @return Store
   */
  [[nodiscard]] Array ranges() const;
  /**
   * @brief Returns the sub-array for characters
   *
   * @return Store
   */
  [[nodiscard]] Array chars() const;

 private:
  friend class Array;

  explicit StringArray(std::shared_ptr<detail::Array> impl);
};

}  // namespace legate

#include "core/data/array.inl"
