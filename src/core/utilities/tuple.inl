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

// Useful for IDEs
#include "core/utilities/tuple.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <type_traits>

namespace legate {

template <typename T>
tuple<T>::tuple(const container_type& values) : data_{values}
{
}

template <typename T>
tuple<T>::tuple(container_type&& values) : data_{std::move(values)}
{
}

template <typename T>
tuple<T>::tuple(std::initializer_list<T> list) : data_{std::move(list)}
{
}

template <typename T>
const T& tuple<T>::at(std::uint32_t idx) const
{
  return data().at(idx);
}

template <typename T>
T& tuple<T>::at(std::uint32_t idx)
{
  return data().at(idx);
}

template <typename T>
const T& tuple<T>::operator[](std::uint32_t idx) const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    return at(idx);
  }
  return data()[idx];
}

template <typename T>
T& tuple<T>::operator[](std::uint32_t idx)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    return at(idx);
  }
  return data()[idx];
}

template <typename T>
bool tuple<T>::operator==(const tuple<T>& other) const
{
  return data() == other.data();
}

template <typename T>
bool tuple<T>::operator!=(const tuple<T>& other) const
{
  return !(*this == other);
}

template <typename T>
bool tuple<T>::operator<(const tuple<T>& other) const
{
  return data() < other.data();
}

template <typename T>
tuple<T> tuple<T>::operator+(const tuple<T>& other) const
{
  return apply(std::plus<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator+(const T& other) const
{
  return apply(std::plus<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator-(const tuple<T>& other) const
{
  return apply(std::minus<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator-(const T& other) const
{
  return apply(std::minus<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator*(const tuple<T>& other) const
{
  return apply(std::multiplies<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator*(const T& other) const
{
  return apply(std::multiplies<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator%(const tuple<T>& other) const
{
  return apply(std::modulus<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator%(const T& other) const
{
  return apply(std::modulus<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator/(const tuple<T>& other) const
{
  return apply(std::divides<>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator/(const T& other) const
{
  return apply(std::divides<>{}, *this, other);
}

template <typename T>
bool tuple<T>::empty() const
{
  return data().empty();
}

template <typename T>
typename tuple<T>::size_type tuple<T>::size() const
{
  return data().size();
}

template <typename T>
void tuple<T>::reserve(size_type size)
{
  return data().reserve(size);
}

template <typename T>
template <typename U>
tuple<T> tuple<T>::insert(std::int32_t pos, U&& value) const
{
  const auto len = static_cast<std::int32_t>(size());
  tuple new_values;

  new_values.reserve(len + 1);
  for (std::int32_t idx = 0; idx < pos; ++idx) {
    new_values.append_inplace(data()[idx]);
  }
  new_values.append_inplace(std::forward<U>(value));
  for (std::int32_t idx = pos; idx < len; ++idx) {
    new_values.append_inplace(data()[idx]);
  }
  return new_values;
}

template <typename T>
template <typename U>
tuple<T> tuple<T>::append(U&& value) const
{
  tuple new_values{data()};

  new_values.append_inplace(std::forward<U>(value));
  return new_values;
}

template <typename T>
tuple<T> tuple<T>::remove(std::int32_t pos) const
{
  tuple new_values;

  if (const auto len = static_cast<std::int32_t>(size())) {
    new_values.reserve(len - 1);
    for (std::int32_t idx = 0; idx < pos; ++idx) {
      new_values.append_inplace(data()[idx]);
    }
    for (std::int32_t idx = pos + 1; idx < len; ++idx) {
      new_values.append_inplace(data()[idx]);
    }
  }
  return new_values;
}

template <typename T>
template <typename U>
tuple<T> tuple<T>::update(std::int32_t pos, U&& value) const
{
  tuple new_values{data()};

  new_values[pos] = std::forward<U>(value);
  return new_values;
}

template <typename T>
template <typename U>
void tuple<T>::insert_inplace(std::int32_t pos, U&& value)
{
  data().insert(begin() + pos, std::forward<U>(value));
}

template <typename T>
template <typename U>
void tuple<T>::append_inplace(U&& value)
{
  data().emplace_back(std::forward<U>(value));
}

template <typename T>
void tuple<T>::remove_inplace(std::int32_t pos)
{
  data().erase(begin() + pos);
}

template <typename T>
template <typename FUNC, typename U>
T tuple<T>::reduce(FUNC&& func, U&& init) const
{
  T agg{std::forward<U>(init)};

  for (auto&& value : data()) {
    agg = func(agg, value);
  }
  return agg;
}

template <typename T>
T tuple<T>::sum() const
{
  return reduce(std::plus<>{}, T{});
}

template <typename T>
T tuple<T>::volume() const
{
  return reduce(std::multiplies<>{}, T{1});
}

template <typename T>
bool tuple<T>::all() const
{
  return all([](auto&& v) { return static_cast<bool>(v); });
}

template <typename T>
template <typename PRED>
bool tuple<T>::all(PRED&& pred) const
{
  return std::all_of(begin(), end(), std::forward<PRED>(pred));
}

template <typename T>
bool tuple<T>::any() const
{
  return any([](auto&& v) { return static_cast<bool>(v); });
}

template <typename T>
template <typename PRED>
bool tuple<T>::any(PRED&& pred) const
{
  return std::any_of(begin(), end(), std::forward<PRED>(pred));
}

template <typename T>
tuple<T> tuple<T>::map(const std::vector<std::int32_t>& mapping) const
{
  tuple new_values;

  new_values.reserve(mapping.size());
  for (auto idx : mapping) {
    new_values.append_inplace(data()[idx]);
  }
  return new_values;
}

template <typename T>
std::string tuple<T>::to_string() const
{
  // clang-tidy says we can make this const, but that's wrong?
  std::stringstream ss;  // NOLINT(misc-const-correctness)

  ss << *this;
  return std::move(ss).str();
}

template <typename U>
std::ostream& operator<<(std::ostream& out, const tuple<U>& tpl)
{
  out << '(';
  std::size_t idx = 0;
  for (auto&& value : tpl) {
    if (idx++ > 0) {
      out << ',';
    }
    out << value;
  }
  out << ')';
  return out;
}

template <typename T>
typename tuple<T>::container_type& tuple<T>::data()
{
  return data_;
}

template <typename T>
const typename tuple<T>::container_type& tuple<T>::data() const
{
  return data_;
}

template <typename T>
typename tuple<T>::iterator tuple<T>::begin()
{
  return data().begin();
}

template <typename T>
typename tuple<T>::const_iterator tuple<T>::cbegin() const
{
  return data().cbegin();
}

template <typename T>
typename tuple<T>::const_iterator tuple<T>::begin() const
{
  return cbegin();
}

template <typename T>
typename tuple<T>::iterator tuple<T>::end()
{
  return data().end();
}

template <typename T>
typename tuple<T>::const_iterator tuple<T>::cend() const
{
  return data().cend();
}

template <typename T>
typename tuple<T>::const_iterator tuple<T>::end() const
{
  return cend();
}

template <typename T>
std::size_t tuple<T>::hash() const
{
  std::size_t result = 0;
  for (auto&& v : data()) {
    hash_combine(result, v);
  }
  return result;
}

// ==========================================================================================

template <typename T>
tuple<T> from_range(T stop)
{
  return from_range(T{}, std::move(stop));
}

template <typename T>
tuple<T> from_range(T start, T stop)
{
  tuple<T> values;

  values.reserve(stop - start);
  for (; start != stop; ++start) {
    values.append_inplace(start);
  }
  return values;
}

template <typename T>
tuple<T> full(detail::type_identity_t<typename tuple<T>::size_type> size, T init)
{
  // Note the use of smooth brackets for initializer! container_type may be an STL container,
  // in which case it suffers from the same 2-argument size-init ctor silently becoming an
  // initializer-list ctor with curly braces.
  typename tuple<T>::container_type cont(size, std::move(init));

  return tuple<T>{std::move(cont)};
}

template <typename FUNC, typename T>
auto apply(FUNC func, const tuple<T>& rhs)
{
  using VAL = std::invoke_result_t<FUNC, T>;
  tuple<VAL> result;

  result.reserve(rhs.size());
  for (auto&& v : rhs) {
    result.append_inplace(func(v));
  }
  return result;
}

template <typename FUNC, typename T1, typename T2>
auto apply(FUNC func, const tuple<T1>& rhs1, const tuple<T2>& rhs2)
{
  using VAL = std::invoke_result_t<FUNC, T1, T2>;
  tuple<VAL> result;

  if (rhs1.size() != rhs2.size()) {
    throw std::invalid_argument{"Operands should have the same size"};
  }
  result.reserve(rhs1.size());
  for (std::uint32_t idx = 0; idx < rhs1.size(); ++idx) {
    result.append_inplace(func(rhs1[idx], rhs2[idx]));
  }
  return result;
}

template <typename FUNC, typename T1, typename T2>
auto apply(FUNC func, const tuple<T1>& rhs1, const T2& rhs2)
{
  using VAL = std::invoke_result_t<FUNC, T1, T2>;
  tuple<VAL> result;

  result.reserve(rhs1.size());
  for (auto&& rhs1_v : rhs1) {
    result.append_inplace(func(rhs1_v, rhs2));
  }
  return result;
}

}  // namespace legate
