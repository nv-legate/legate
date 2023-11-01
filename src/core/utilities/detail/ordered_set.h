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

#include <unordered_set>
#include <vector>

namespace legate {

// A set implementation that keeps the elements in the order in which they are inserted.
// Internally it uses a vector and a set, the latter of which is used for deduplication.
// This container is not very efficient if used with non-pointer data types

template <typename T>
class ordered_set {
 public:
  void insert(const T& value);
  void insert(T&& value);
  [[nodiscard]] const std::vector<T>& elements() const;

 private:
  std::vector<T> elements_{};
  std::unordered_set<T> element_set_{};
};

}  // namespace legate

#include "core/utilities/detail/ordered_set.inl"
