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
  tuple(const tuple<T>&)            = default;
  tuple(tuple<T>&&)                 = default;
  tuple& operator=(const tuple<T>&) = default;
  tuple& operator=(tuple<T>&&)      = default;

 public:
  const T& operator[](uint32_t idx) const;
  T& operator[](uint32_t idx);

 public:
  bool operator==(const tuple<T>& other) const;
  bool operator!=(const tuple<T>& other) const;
  bool operator<(const tuple<T>& other) const;
  tuple<T> operator+(const tuple<T>& other) const;
  tuple<T> operator+(const T& other) const;
  tuple<T> operator-(const tuple<T>& other) const;
  tuple<T> operator-(const T& other) const;
  tuple<T> operator*(const tuple<T>& other) const;
  tuple<T> operator*(const T& other) const;
  tuple<T> operator%(const tuple<T>& other) const;
  tuple<T> operator%(const T& other) const;
  tuple<T> operator/(const tuple<T>& other) const;
  tuple<T> operator/(const T& other) const;

 public:
  bool empty() const;
  size_t size() const;

 public:
  tuple<T> insert(int32_t pos, const T& value) const;
  tuple<T> append(const T& value) const;
  tuple<T> remove(int32_t pos) const;
  tuple<T> update(int32_t pos, const T& value) const;

  void insert_inplace(int32_t pos, const T& value);
  void append_inplace(const T& value);
  void remove_inplace(int32_t pos);

 public:
  template <typename FUNC>
  T reduce(FUNC func, const T& init) const;
  T sum() const;
  T volume() const;
  bool all() const;
  template <typename PRED>
  bool all(PRED pred) const;
  bool any() const;
  template <typename PRED>
  bool any(PRED pred) const;
  tuple<T> map(const std::vector<int32_t>& mapping) const;

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

template <typename FUNC, typename T>
auto apply(FUNC func, const tuple<T>& rhs);

template <typename FUNC, typename T1, typename T2>
auto apply(FUNC func, const tuple<T1>& rhs1, const tuple<T2>& rhs2);

template <typename FUNC, typename T1, typename T2>
auto apply(FUNC func, const tuple<T1>& rhs1, const T2& rhs2);

}  // namespace legate

#include "core/utilities/tuple.inl"
