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

#include "core/utilities/hash.h"

#include "legate_defines.h"

#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <vector>

namespace legate {

// A simple wrapper around an STL vector to provide common utilities
template <typename T>
class tuple {
 public:
  using container_type = std::vector<T>;
  using value_type     = typename container_type::value_type;
  using size_type      = typename container_type::size_type;
  using iterator       = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  tuple() noexcept = default;

  explicit tuple(const container_type& values);
  explicit tuple(container_type&& values);
  tuple(std::initializer_list<T> list);

  tuple(const tuple&)                = default;
  tuple(tuple&&) noexcept            = default;
  tuple& operator=(const tuple&)     = default;
  tuple& operator=(tuple&&) noexcept = default;

  [[nodiscard]] const T& at(uint32_t idx) const;
  [[nodiscard]] T& at(uint32_t idx);
  [[nodiscard]] const T& operator[](uint32_t idx) const;
  [[nodiscard]] T& operator[](uint32_t idx);

  bool operator==(const tuple& other) const;
  bool operator!=(const tuple& other) const;
  bool operator<(const tuple& other) const;
  tuple operator+(const tuple& other) const;
  tuple operator+(const T& other) const;
  tuple operator-(const tuple& other) const;
  tuple operator-(const T& other) const;
  tuple operator*(const tuple& other) const;
  tuple operator*(const T& other) const;
  tuple operator%(const tuple& other) const;
  tuple operator%(const T& other) const;
  tuple operator/(const tuple& other) const;
  tuple operator/(const T& other) const;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] size_type size() const;
  void reserve(size_type size);

  template <typename U = T>
  [[nodiscard]] tuple insert(int32_t pos, U&& value) const;
  template <typename U = T>
  [[nodiscard]] tuple append(U&& value) const;
  [[nodiscard]] tuple remove(int32_t pos) const;
  template <typename U = T>
  [[nodiscard]] tuple update(int32_t pos, U&& value) const;

  template <typename U = T>
  void insert_inplace(int32_t pos, U&& value);
  template <typename U = T>
  void append_inplace(U&& value);
  void remove_inplace(int32_t pos);

  template <typename FUNC, typename U>
  [[nodiscard]] T reduce(FUNC&& func, U&& init) const;
  [[nodiscard]] T sum() const;
  [[nodiscard]] T volume() const;
  [[nodiscard]] bool all() const;
  template <typename PRED>
  [[nodiscard]] bool all(PRED&& pred) const;
  [[nodiscard]] bool any() const;
  template <typename PRED>
  [[nodiscard]] bool any(PRED&& pred) const;
  [[nodiscard]] tuple map(const std::vector<int32_t>& mapping) const;

  [[nodiscard]] std::string to_string() const;
  template <typename U>
  friend std::ostream& operator<<(std::ostream& out, const tuple<U>& tpl);

  [[nodiscard]] container_type& data();
  [[nodiscard]] const container_type& data() const;

  [[nodiscard]] iterator begin();
  [[nodiscard]] const_iterator cbegin() const;
  [[nodiscard]] const_iterator begin() const;

  [[nodiscard]] iterator end();
  [[nodiscard]] const_iterator cend() const;
  [[nodiscard]] const_iterator end() const;

  [[nodiscard]] size_t hash() const;

 private:
  container_type data_{};
};

template <typename T>
[[nodiscard]] tuple<T> from_range(T stop);

template <typename T>
[[nodiscard]] tuple<T> from_range(T start, T stop);

namespace detail {

template <typename T>
struct type_identity {
  using type = T;
};

template <typename T>
using type_identity_t = typename type_identity<T>::type;

}  // namespace detail

template <typename T>
[[nodiscard]] tuple<T> full(detail::type_identity_t<typename tuple<T>::size_type> size, T init);

template <typename FUNC, typename T>
[[nodiscard]] auto apply(FUNC func, const tuple<T>& rhs);

template <typename FUNC, typename T1, typename T2>
[[nodiscard]] auto apply(FUNC func, const tuple<T1>& rhs1, const tuple<T2>& rhs2);

template <typename FUNC, typename T1, typename T2>
[[nodiscard]] auto apply(FUNC func, const tuple<T1>& rhs1, const T2& rhs2);

}  // namespace legate

#include "core/utilities/tuple.inl"
