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

#include "core/utilities/typedefs.h"

namespace legate::mapping::detail {
class Store;
}  // namespace legate::mapping::detail

namespace legate::mapping {

/**
 * @ingroup mapping
 * @brief A metadata class that mirrors the structure of legate::Store but contains
 * only the data relevant to mapping
 */
class Store {
 public:
  /**
   * @brief Indicates whether the store is backed by a future
   *
   * @return true The store is backed by a future
   * @return false The store is backed by a region field
   */
  [[nodiscard]] bool is_future() const;
  /**
   * @brief Indicates whether the store is unbound
   *
   * @return true The store is unbound
   * @return false The store is a normal store
   */
  [[nodiscard]] bool unbound() const;
  /**
   * @brief Returns the store's dimension
   *
   * @return Store's dimension
   */
  [[nodiscard]] int32_t dim() const;

  /**
   * @brief Indicates whether the store is a reduction store
   *
   * @return true The store is a reduction store
   * @return false The store is either an input or output store
   */
  [[nodiscard]] bool is_reduction() const;
  /**
   * @brief Returns the reduction operator id for the store
   *
   * @return Reduction oeprator id
   */
  [[nodiscard]] int32_t redop() const;

  /**
   * @brief Indicates whether the store can colocate in an instance with a given store
   *
   * @param other Store against which the colocation is checked
   *
   * @return true The store can colocate with the input
   * @return false The store cannot colocate with the input
   */
  [[nodiscard]] bool can_colocate_with(const Store& other) const;

  /**
   * @brief Returns the store's domain
   *
   * @return Store's domain
   */
  template <int32_t DIM>
  [[nodiscard]] Rect<DIM> shape() const;
  /**
   * @brief Returns the store's domain in a dimension-erased domain type
   *
   * @return Store's domain in a dimension-erased domain type
   */
  [[nodiscard]] Domain domain() const;

  [[nodiscard]] const detail::Store* impl() const noexcept;

  explicit Store(const detail::Store* impl) noexcept;

  Store() = default;

 private:
  const detail::Store* impl_{};
};

}  // namespace legate::mapping

#include "core/mapping/store.inl"
