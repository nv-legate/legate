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

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

namespace legate {

// A simple wrapper around an STL vector to provide common utilities
template <typename T>
class tuple {
 public:
  tuple();
  tuple(const std::vector<T>& values);
  tuple(std::vector<T>&& values);
  tuple(std::initializer_list<T> list);
  tuple(size_t size, T init);

 public:
  tuple(const tuple&) = default;
  tuple(tuple&&)      = default;
  tuple& operator=(const tuple&) = default;
  tuple& operator=(tuple&&) = default;

 public:
  const T& operator[](uint32_t idx) const;
  T& operator[](uint32_t idx);

 public:
  bool empty() const;
  size_t size() const;

 public:
  tuple<T> insert(int32_t pos, const T& value) const;
  tuple<T> append(const T& value) const;
  tuple<T> remove(int32_t pos) const;

  void insert_inplace(int32_t pos, const T& value);
  void append_inplace(const T& value);
  void remove_inplace(int32_t pos);

 public:
  template <typename FUNC>
  T reduce(FUNC func, const T& init) const;
  T volume() const;
  template <typename PRED>
  bool all(PRED pred) const;
  template <typename PRED>
  bool any(PRED pred) const;

 public:
  std::string to_string() const;
  template <typename _T>
  friend std::ostream& operator<<(std::ostream& out, const tuple<_T>& tpl);

 public:
  const std::vector<T>& data() const;

 private:
  std::vector<T> data_{};
};

template <typename T>
tuple<T> from_range(T stop);

template <typename T>
tuple<T> from_range(T start, T stop);

}  // namespace legate

#include "core/utilities/tuple.inl"
