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

#include "core/data/physical_store.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

#include <cstdint>
#include <memory>

/**
 * @file
 * @brief Class definition for legate::PhysicalArray
 */

namespace legate {

namespace detail {
struct PhysicalArray;
}  // namespace detail

class ListPhysicalArray;
class StringPhysicalArray;

class PhysicalArray {
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
   * @return Physical store
   *
   * @throw std::invalid_argument If the array is not a base array
   */
  [[nodiscard]] PhysicalStore data() const;
  /**
   * @brief Returns the store containing the array's null mask
   *
   * @return Physical store
   *
   * @throw std::invalid_argument If the array is not nullable
   */
  [[nodiscard]] PhysicalStore null_mask() const;
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
  [[nodiscard]] PhysicalArray child(uint32_t index) const;

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
  [[nodiscard]] ListPhysicalArray as_list_array() const;
  /**
   * @brief Casts this array as a string array
   *
   * @return String array
   *
   * @throw std::invalid_argument If the array is not a string array
   */
  [[nodiscard]] StringPhysicalArray as_string_array() const;

  explicit PhysicalArray(std::shared_ptr<detail::PhysicalArray> impl);

  [[nodiscard]] const std::shared_ptr<detail::PhysicalArray>& impl() const;

  PhysicalArray()                                    = default;
  virtual ~PhysicalArray() noexcept                  = default;
  PhysicalArray(const PhysicalArray&) noexcept       = default;
  PhysicalArray& operator=(const PhysicalArray&)     = default;
  PhysicalArray(PhysicalArray&&) noexcept            = default;
  PhysicalArray& operator=(PhysicalArray&&) noexcept = default;

 private:
  void check_shape_dimension(int32_t dim) const;

 protected:
  std::shared_ptr<detail::PhysicalArray> impl_{};
};

class ListPhysicalArray : public PhysicalArray {
 public:
  /**
   * @brief Returns the sub-array for descriptors
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray descriptor() const;
  /**
   * @brief Returns the sub-array for variable size data
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray vardata() const;

 private:
  friend class PhysicalArray;

  explicit ListPhysicalArray(std::shared_ptr<detail::PhysicalArray> impl);
};

class StringPhysicalArray : public PhysicalArray {
 public:
  /**
   * @brief Returns the sub-array for ranges
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray ranges() const;
  /**
   * @brief Returns the sub-array for characters
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray chars() const;

 private:
  friend class PhysicalArray;

  explicit StringPhysicalArray(std::shared_ptr<detail::PhysicalArray> impl);
};

}  // namespace legate

#include "core/data/physical_array.inl"
