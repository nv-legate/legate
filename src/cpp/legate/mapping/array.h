/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/store.h>
#include <legate/type/types.h>

namespace legate::mapping::detail {
class Array;
}  // namespace legate::mapping::detail

namespace legate::mapping {

class Array {
 public:
  /**
   * @brief Indicates if the array is nullable
   *
   * @return true If the array is nullable
   * @return false Otherwise
   */
  [[nodiscard]] bool nullable() const;
  /**
   * @brief Returns the dimension of the array
   *
   * @return Array's dimension
   */
  [[nodiscard]] std::int32_t dim() const;
  /**
   * @brief Returns the array's type
   *
   * @return Type
   */
  [[nodiscard]] Type type() const;
  /**
   * @brief Returns metadata of the store containing the array's data
   *
   * @return Store metadata
   */
  [[nodiscard]] Store data() const;
  /**
   * @brief Returns metadata of the store containing the array's null mask
   *
   * @return Store metadata
   *
   * @throw std::invalid_argument If the array is not nullable
   */
  [[nodiscard]] Store null_mask() const;
  /**
   * @brief Returns metadata of all stores associated with this array
   *
   * @return Vector of store metadata
   */
  [[nodiscard]] std::vector<Store> stores() const;
  /**
   * @brief Returns the array's domain
   *
   * @return Array's domain
   */
  template <std::int32_t DIM>
  [[nodiscard]] Rect<DIM> shape() const;
  /**
   * @brief Returns the array's domain in a dimension-erased domain type
   *
   * @return Array's domain in a dimension-erased domain type
   */
  [[nodiscard]] Domain domain() const;

  explicit Array(const detail::Array* impl);

 private:
  const detail::Array* impl_{};
};

}  // namespace legate::mapping

#include <legate/mapping/array.inl>
