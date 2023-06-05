/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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

 private:
  std::map<T, size_t> map_;
};

}  // namespace legate

#include "core/utilities/multi_set.inl"
