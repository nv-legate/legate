/* Copyright 2021 NVIDIA Corporation
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

#include <unordered_set>
#include <vector>

namespace legate {

// A set implementation that keeps the elements in the order in which they are inserted.
// Internally it uses a vector and a set, the latter of which is used for deduplication.
// This container is not very efficient if used with non-pointer data types

template <typename T>
class ordered_set {
 public:
  ordered_set() {}

 public:
  void insert(const T& value)
  {
    if (element_set_.find(value) != element_set_.end()) return;
    elements_.push_back(value);
    element_set_.insert(value);
  }

 public:
  const std::vector<T>& elements() const { return elements_; }

 private:
  std::vector<T> elements_{};
  std::unordered_set<T> element_set_{};
};

}  // namespace legate
