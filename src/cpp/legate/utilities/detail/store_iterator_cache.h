/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <legate/utilities/detail/store_iterator_cache.inl>
