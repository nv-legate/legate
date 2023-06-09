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

// Useful for IDEs
#include "core/utilities/tuple.h"

#include <numeric>
#include <type_traits>
#include <vector>

namespace legate {

template <typename T>
tuple<T>::tuple()
{
}

template <typename T>
tuple<T>::tuple(const std::vector<T>& values) : data_(values)
{
}

template <typename T>
tuple<T>::tuple(std::vector<T>&& values) : data_(std::move(values))
{
}

template <typename T>
tuple<T>::tuple(std::initializer_list<T> list) : data_(list)
{
}

template <typename T>
tuple<T>::tuple(size_t size, T init) : data_(size, init)
{
}

template <typename T>
const T& tuple<T>::operator[](uint32_t idx) const
{
  return data_[idx];
}

template <typename T>
T& tuple<T>::operator[](uint32_t idx)
{
  return data_[idx];
}

template <typename T>
bool tuple<T>::operator==(const tuple<T>& other) const
{
  return data_ == other.data_;
}

template <typename T>
bool tuple<T>::operator!=(const tuple<T>& other) const
{
  return data_ != other.data_;
}

template <typename T>
bool tuple<T>::operator<(const tuple<T>& other) const
{
  return data_ < other.data_;
}

template <typename T>
tuple<T> tuple<T>::operator+(const tuple<T>& other) const
{
  return apply(std::plus<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator+(const T& other) const
{
  return apply(std::plus{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator-(const tuple<T>& other) const
{
  return apply(std::minus<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator-(const T& other) const
{
  return apply(std::minus<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator*(const tuple<T>& other) const
{
  return apply(std::multiplies<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator*(const T& other) const
{
  return apply(std::multiplies<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator%(const tuple<T>& other) const
{
  return apply(std::modulus<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator%(const T& other) const
{
  return apply(std::modulus<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator/(const tuple<T>& other) const
{
  return apply(std::divides<T>{}, *this, other);
}

template <typename T>
tuple<T> tuple<T>::operator/(const T& other) const
{
  return apply(std::divides<T>{}, *this, other);
}

template <typename T>
bool tuple<T>::empty() const
{
  return data_.empty();
}

template <typename T>
size_t tuple<T>::size() const
{
  return data_.size();
}

template <typename T>
tuple<T> tuple<T>::insert(int32_t pos, const T& value) const
{
  std::vector<T> new_values;
  auto len = static_cast<int32_t>(data_.size());

  for (int32_t idx = 0; idx < pos; ++idx) new_values.push_back(data_[idx]);
  new_values.push_back(value);
  for (int32_t idx = pos; idx < len; ++idx) new_values.push_back(data_[idx]);
  return tuple<T>(std::move(new_values));
}

template <typename T>
tuple<T> tuple<T>::append(const T& value) const
{
  std::vector<T> new_values(data_);
  new_values.push_back(value);
  return tuple<T>(std::move(new_values));
}

template <typename T>
tuple<T> tuple<T>::remove(int32_t pos) const
{
  std::vector<T> new_values;
  auto len = static_cast<int32_t>(data_.size());

  for (int32_t idx = 0; idx < pos; ++idx) new_values.push_back(data_[idx]);
  for (int32_t idx = pos + 1; idx < len; ++idx) new_values.push_back(data_[idx]);
  return tuple<T>(std::move(new_values));
}

template <typename T>
tuple<T> tuple<T>::update(int32_t pos, const T& value) const
{
  std::vector<T> new_values(data_);
  new_values[pos] = value;
  return tuple<T>(std::move(new_values));
}

template <typename T>
void tuple<T>::insert_inplace(int32_t pos, const T& value)
{
  auto oldlen = static_cast<int32_t>(data_.size());
  data_.resize(oldlen + 1);

  for (int32_t idx = oldlen; idx > pos; --idx) data_[idx + 1] = data_[idx];
  data_[pos] = value;
}

template <typename T>
void tuple<T>::append_inplace(const T& value)
{
  data_.push_back(value);
}

template <typename T>
void tuple<T>::remove_inplace(int32_t pos)
{
  data_.erase(data_.begin() + pos);
}

template <typename T>
template <typename FUNC>
T tuple<T>::reduce(FUNC func, const T& init) const
{
  T agg{init};
  for (auto value : data_) agg = func(agg, value);
  return agg;
}

template <typename T>
T tuple<T>::sum() const
{
  return reduce(std::plus{}, T(0));
}

template <typename T>
T tuple<T>::volume() const
{
  return reduce(std::multiplies<T>{}, T{1});
}

template <typename T>
bool tuple<T>::all() const
{
  return all([](auto v) { return static_cast<bool>(v); });
}

template <typename T>
template <typename PRED>
bool tuple<T>::all(PRED pred) const
{
  return std::all_of(data_.begin(), data_.end(), pred);
}

template <typename T>
bool tuple<T>::any() const
{
  return any([](auto v) { return static_cast<bool>(v); });
}

template <typename T>
template <typename PRED>
bool tuple<T>::any(PRED pred) const
{
  return std::any_of(data_.begin(), data_.end(), pred);
}

template <typename T>
tuple<T> tuple<T>::map(const std::vector<int32_t>& mapping) const
{
  std::vector<T> new_values;
  for (auto idx : mapping) new_values.push_back(data_[idx]);
  return tuple<T>(std::move(new_values));
}

template <typename T>
std::string tuple<T>::to_string() const
{
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

template <typename _T>
std::ostream& operator<<(std::ostream& out, const tuple<_T>& tpl)
{
  out << "(";
  for (auto value : tpl.data_) out << value << ",";
  out << ")";
  return out;
}

template <typename T>
const std::vector<T>& tuple<T>::data() const
{
  return data_;
}

template <typename T>
tuple<T> from_range(T stop)
{
  return from_range(T{0}, stop);
}

template <typename T>
tuple<T> from_range(T start, T stop)
{
  std::vector<T> values(stop - start);
  std::iota(values.begin(), values.end(), start);
  return tuple<T>(std::move(values));
}

template <typename FUNC, typename T>
auto apply(FUNC func, const tuple<T>& rhs)
{
  using VAL = typename std::invoke_result<FUNC, T>::type;
  std::vector<VAL> result;
  for (auto& v : rhs.data()) result.push_back(func(v));
  return tuple<VAL>(std::move(result));
}

template <typename FUNC, typename T1, typename T2>
auto apply(FUNC func, const tuple<T1>& rhs1, const tuple<T2>& rhs2)
{
  if (rhs1.size() != rhs2.size()) throw std::invalid_argument("Operands should have the same size");
  using VAL = typename std::invoke_result<FUNC, T1, T2>::type;
  std::vector<VAL> result;
  for (uint32_t idx = 0; idx < rhs1.size(); ++idx) result.push_back(func(rhs1[idx], rhs2[idx]));
  return tuple<VAL>(std::move(result));
}

template <typename FUNC, typename T1, typename T2>
auto apply(FUNC func, const tuple<T1>& rhs1, const T2& rhs2)
{
  using VAL = typename std::invoke_result<FUNC, T1, T2>::type;
  std::vector<VAL> result;
  for (uint32_t idx = 0; idx < rhs1.size(); ++idx) result.push_back(func(rhs1[idx], rhs2));
  return tuple<VAL>(std::move(result));
}

}  // namespace legate
