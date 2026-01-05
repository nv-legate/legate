/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>
#include <vector>

namespace legate {

// A set implementation that keeps the elements in the order in which they are inserted.
// Internally it uses a vector and a set, the latter of which is used for deduplication.
// This container is not very efficient if used with non-pointer data types

template <typename T>
class ordered_set {  // NOLINT(readability-identifier-naming)
 public:
  void insert(const T& value);
  void insert(T&& value);
  [[nodiscard]] const std::vector<T>& elements() const;

 private:
  std::vector<T> elements_{};
  std::unordered_set<T> element_set_{};
};

}  // namespace legate

#include <legate/utilities/detail/ordered_set.inl>
