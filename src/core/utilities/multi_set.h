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

#include <map>

namespace legate {

/**
 * @brief A set variant that allows for multiple instances of each element.
 */
template <typename T>
class MultiSet {
 public:
  MultiSet() {}

 public:
  /**
   * @brief Add a value to the container.
   */
  void add(const T& value);

  /**
   * @brief Remove an instance of a value from the container (other instances might still remain).
   *
   * @return Whether this removed the last instance of the value.
   */
  bool remove(const T& value);

  /**
   * @brief Test whether a value is present in the container (at least once).
   */
  bool contains(const T& value) const;

  /**
   * @brief Clears the container
   */
  void clear();

 private:
  std::map<T, size_t> map_;
};

}  // namespace legate

#include "core/utilities/multi_set.inl"
