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

#include "core/mapping/store.h"
#include "core/type/type_info.h"

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
  [[nodiscard]] int32_t dim() const;
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
   * @brief Returns metadat of all stores associated with this array
   *
   * @return Vector of store metadata
   */
  [[nodiscard]] std::vector<Store> stores() const;
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

  explicit Array(const detail::Array* impl);

 private:
  const detail::Array* impl_{};
};

}  // namespace legate::mapping

#include "core/mapping/array.inl"
