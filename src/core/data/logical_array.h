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

#include <memory>

#include "core/data/array.h"
#include "core/data/logical_store.h"
#include "core/data/shape.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition for legate::LogicalArray
 */

namespace legate::detail {

struct LogicalArray;

}  // namespace legate::detail

namespace legate {

class ListLogicalArray;
class StringLogicalArray;

/**
 * @ingroup data
 *
 * @brief A multi-dimensional array
 */
class LogicalArray {
 public:
  /**
   * @brief Returns the number of dimensions of the array.
   *
   * @return The number of dimensions
   */
  int32_t dim() const;
  /**
   * @brief Returns the element type of the array.
   *
   * @return Type of elements in the store
   */
  Type type() const;
  /**
   * @brief Returns the shape of the array.
   *
   * Flushes the scheduling window if the store is unbound and has no shape assigned.
   *
   * @return The store's shape
   */
  const Shape& extents() const;
  /**
   * @brief Returns the number of elements in the array.
   *
   * Flushes the scheduling window if the store is unbound and has no shape assigned.
   *
   * @return The number of elements in the store
   */
  size_t volume() const;
  /**
   * @brief Indicates whether the array is unbound
   *
   * @return true The array is unbound
   * @return false The array is normal
   */
  bool unbound() const;
  /**
   * @brief Indicates whether the array is nullable
   *
   * @return true The array is nullable
   * @return false The array is non-nullable
   */
  bool nullable() const;
  /**
   * @brief Indicates whether the array has child arrays
   *
   * @return true The array has child arrays
   * @return false Otherwise
   */
  bool nested() const;
  /**
   * @brief Returns the number of child sub-arrays
   *
   * @return Number of child sub-arrays
   */
  uint32_t num_children() const;

 public:
  /**
   * @brief Adds an extra dimension to the array.
   *
   * @param extra_dim Position for a new dimension
   * @param dim_size Extent of the new dimension
   *
   * @return A new array with an extra dimension
   *
   * @throw std::invalid_argument When `extra_dim` is not a valid dimension name
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  LogicalArray promote(int32_t extra_dim, size_t dim_size) const;
  /**
   * @brief Projects out a dimension of the array.
   *
   * @param dim Dimension to project out
   * @param index Index on the chosen dimension
   *
   * @return A new array with one fewer dimension
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name or `index` is out of bounds
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  LogicalArray project(int32_t dim, int64_t index) const;
  /**
   * @brief Slices a contiguous sub-section of the array.
   *
   * @param dim Dimension to slice
   * @param sl Slice descriptor
   *
   * @return A new array that corresponds to the sliced section
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  LogicalArray slice(int32_t dim, Slice sl) const;
  /**
   * @brief Reorders dimensions of the array.
   *
   * @param axes Mapping from dimensions of the resulting array to those of the input
   *
   * @return A new array with the dimensions transposed
   *
   * @throw std::invalid_argument If any of the following happens: 1) The length of `axes` doesn't
   * match the array's dimension; 2) `axes` has duplicates; 3) Any axis in `axes` is an invalid
   * axis name.
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  LogicalArray transpose(const std::vector<int32_t>& axes) const;
  /**
   * @brief Delinearizes a dimension into multiple dimensions.
   *
   * @param dim Dimension to delinearize
   * @param sizes Extents for the resulting dimensions
   *
   * @return A new array with the chosen dimension delinearized
   *
   * @throw std::invalid_argument If `dim` is invalid for the array or `sizes` does not preserve
   * the extent of the chosen dimenison
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  LogicalArray delinearize(int32_t dim, const std::vector<int64_t>& sizes) const;

 public:
  /**
   * @brief Returns the store of this array
   *
   * @return Logical store
   */
  LogicalStore data() const;
  /**
   * @brief Returns the store of this array
   *
   * @return Logical store
   */
  LogicalStore null_mask() const;
  /**
   * @brief Returns the sub-array of a given index
   *
   * @param index Sub-array index
   *
   * @return Logical array
   *
   * @throw std::invalid_argument If the array has no child arrays, or the array is an unbound
   * struct array
   * @throw std::out_of_range If the index is out of range
   */
  LogicalArray child(uint32_t index) const;

 public:
  /**
   * @brief Creates a physical array for this logical array
   *
   * This call blocks the client's control flow and fetches the data for the whole array to the
   * current node
   *
   * @return A physical array of the logical array
   */
  Array get_physical_array() const;

 public:
  /**
   * @brief Casts this array as a list array
   *
   * @return List array
   *
   * @throw std::invalid_argument If the array is not a list array
   */
  ListLogicalArray as_list_array() const;
  /**
   * @brief Casts this array as a string array
   *
   * @return String array
   *
   * @throw std::invalid_argument If the array is not a string array
   */
  StringLogicalArray as_string_array() const;

 public:
  LogicalArray() = default;
  LogicalArray(std::shared_ptr<detail::LogicalArray> impl);
  virtual ~LogicalArray();

 public:
  LogicalArray(const LogicalArray& other)            = default;
  LogicalArray& operator=(const LogicalArray& other) = default;
  LogicalArray(LogicalArray&& other)                 = default;
  LogicalArray& operator=(LogicalArray&& other)      = default;

 public:
  LogicalArray(LogicalStore store);
  LogicalArray(LogicalStore store, LogicalStore null_mask);

 public:
  std::shared_ptr<detail::LogicalArray> impl() const { return impl_; }

 protected:
  std::shared_ptr<detail::LogicalArray> impl_{nullptr};
};

class ListLogicalArray : public LogicalArray {
 public:
  /**
   * @brief Returns the sub-array for descriptors
   *
   * @return Store
   */
  LogicalArray descriptor() const;
  /**
   * @brief Returns the sub-array for variable size data
   *
   * @return Store
   */
  LogicalArray vardata() const;

 private:
  friend class LogicalArray;
  ListLogicalArray(std::shared_ptr<detail::LogicalArray> impl);
};

class StringLogicalArray : public LogicalArray {
 public:
  /**
   * @brief Returns the sub-array for offsets
   *
   * @return Store
   */
  LogicalArray offsets() const;
  /**
   * @brief Returns the sub-array for characters
   *
   * @return Store
   */
  LogicalArray chars() const;

 private:
  friend class LogicalArray;
  StringLogicalArray(std::shared_ptr<detail::LogicalArray> impl);
};

}  // namespace legate
