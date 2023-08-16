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
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition for legate::Array
 */

namespace legate {

namespace detail {
class Array;
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
  bool nullable() const;
  /**
   * @brief Returns the dimension of the array
   *
   * @return Array's dimension
   */
  int32_t dim() const;
  /**
   * @brief Returns the array's type
   *
   * @return Type
   */
  Type type() const;
  /**
   * @brief Indicates if the array has child arrays
   *
   * @return true If the array has child arrays
   * @return false Otherwise
   */
  bool nested() const;

 public:
  /**
   * @brief Returns the store containing the array's data
   *
   * @return Store
   *
   * @throw std::invalid_argument If the array is not a base array
   */
  Store data() const;
  /**
   * @brief Returns the store containing the array's null mask
   *
   * @return Store
   *
   * @throw std::invalid_argument If the array is not nullable
   */
  Store null_mask() const;
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
  Array child(uint32_t index) const;

 public:
  /**
   * @brief Returns the array's domain
   *
   * @return Array's domain
   */
  template <int32_t DIM>
  Rect<DIM> shape() const;
  /**
   * @brief Returns the array's domain in a dimension-erased domain type
   *
   * @return Array's domain in a dimension-erased domain type
   */
  Domain domain() const;

 public:
  /**
   * @brief Casts this array as a list array
   *
   * @return List array
   *
   * @throw std::invalid_argument If the array is not a list array
   */
  ListArray as_list_array() const;
  /**
   * @brief Casts this array as a string array
   *
   * @return String array
   *
   * @throw std::invalid_argument If the array is not a string array
   */
  StringArray as_string_array() const;

 private:
  void check_shape_dimension(const int32_t dim) const;

 public:
  Array(std::shared_ptr<detail::Array> impl);
  std::shared_ptr<detail::Array> impl() const { return impl_; }

 public:
  Array(const Array&);
  Array& operator=(const Array&);
  Array(Array&&);
  Array& operator=(Array&&);

 public:
  virtual ~Array();

 protected:
  std::shared_ptr<detail::Array> impl_{nullptr};
};

class ListArray : public Array {
 public:
  /**
   * @brief Returns the sub-array for descriptors
   *
   * @return Store
   */
  Array descriptor() const;
  /**
   * @brief Returns the sub-array for variable size data
   *
   * @return Store
   */
  Array vardata() const;

 private:
  friend class Array;
  ListArray(std::shared_ptr<detail::Array> impl);
};

class StringArray : public Array {
 public:
  /**
   * @brief Returns the sub-array for ranges
   *
   * @return Store
   */
  Array ranges() const;
  /**
   * @brief Returns the sub-array for characters
   *
   * @return Store
   */
  Array chars() const;

 private:
  friend class Array;
  StringArray(std::shared_ptr<detail::Array> impl);
};

}  // namespace legate

#include "core/data/array.inl"
