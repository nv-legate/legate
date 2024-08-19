/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>

namespace legate::detail {

template <typename T>
class StoreIteratorCache {
 public:
  using container_type = std::vector<T>;

  template <typename U>
  [[nodiscard]] container_type& operator()(const U& array);
  template <typename U>
  [[nodiscard]] const container_type& operator()(const U& array) const;

 private:
  mutable container_type cache_{};
};

}  // namespace legate::detail

#include "core/utilities/detail/store_iterator_cache.inl"
