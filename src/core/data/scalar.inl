/* Copyright 2021-2022 NVIDIA Corporation
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

#include <string_view>

// Useful for IDEs
#include "core/data/scalar.h"

namespace legate {

template <typename T>
Scalar::Scalar(T value) : impl_(create_impl(primitive_type(legate_type_code_of<T>), &value, true))
{
  static_assert(legate_type_code_of<T> != Type::Code::FIXED_ARRAY);
  static_assert(legate_type_code_of<T> != Type::Code::STRUCT);
  static_assert(legate_type_code_of<T> != Type::Code::STRING);
  static_assert(legate_type_code_of<T> != Type::Code::INVALID);
}

template <typename T>
Scalar::Scalar(T value, Type type) : impl_(create_impl(type, &value, true))
{
  if (type.code() == Type::Code::INVALID)
    throw std::invalid_argument("Invalid type cannot be used");
  if (type.size() != sizeof(T))
    throw std::invalid_argument("Size of the value doesn't match with the type");
}

template <typename T>
Scalar::Scalar(const std::vector<T>& values)
  : impl_(create_impl(
      fixed_array_type(primitive_type(legate_type_code_of<T>), values.size()), values.data(), true))
{
}

template <int32_t DIM>
Scalar::Scalar(const Point<DIM>& point) : impl_(create_impl(point_type(DIM), &point, true))
{
}

template <int32_t DIM>
Scalar::Scalar(const Rect<DIM>& rect) : impl_(create_impl(rect_type(DIM), &rect, true))
{
}

template <typename VAL>
VAL Scalar::value() const
{
  auto ty = type();
  if (ty.code() == Type::Code::STRING)
    throw std::invalid_argument("String cannot be casted to other types");
  if (sizeof(VAL) != ty.size())
    throw std::invalid_argument("Size of the scalar is " + std::to_string(ty.size()) +
                                ", but the requested type has size " + std::to_string(sizeof(VAL)));
  return *static_cast<const VAL*>(ptr());
}

template <>
inline std::string Scalar::value() const
{
  if (type().code() != Type::Code::STRING)
    throw std::invalid_argument("Type of the scalar is not string");
  const void* data  = ptr();
  auto len          = *static_cast<const uint32_t*>(data);
  const auto* begin = static_cast<const char*>(data) + sizeof(uint32_t);
  const auto* end   = begin + len;
  return std::string(begin, end);
}

template <>
inline std::string_view Scalar::value() const
{
  if (type().code() != Type::Code::STRING)
    throw std::invalid_argument("Type of the scalar is not string");
  const void* data  = ptr();
  auto len          = *static_cast<const uint32_t*>(data);
  const auto* begin = static_cast<const char*>(data) + sizeof(uint32_t);
  return std::string_view(begin, len);
}

template <typename VAL>
Span<const VAL> Scalar::values() const
{
  auto ty = type();
  if (ty.code() == Type::Code::FIXED_ARRAY) {
    auto arr_type  = ty.as_fixed_array_type();
    auto elem_type = arr_type.element_type();
    if (sizeof(VAL) != elem_type.size())
      throw std::invalid_argument(
        "The scalar's element type has size " + std::to_string(elem_type.size()) +
        ", but the requested element type has size " + std::to_string(sizeof(VAL)));
    auto size = arr_type.num_elements();
    return Span<const VAL>(reinterpret_cast<const VAL*>(ptr()), size);
  } else if (ty.code() == Type::Code::STRING) {
    if (sizeof(VAL) != 1)
      throw std::invalid_argument(
        "String scalar can only be converted into a span of a type whose size is 1 byte");
    auto data         = ptr();
    auto len          = *static_cast<const uint32_t*>(data);
    const auto* begin = static_cast<const char*>(data) + sizeof(uint32_t);
    return Span<const VAL>(reinterpret_cast<const VAL*>(begin), len);
  } else {
    if (sizeof(VAL) != ty.size())
      throw std::invalid_argument("Size of the scalar is " + std::to_string(ty.size()) +
                                  ", but the requested element type has size " +
                                  std::to_string(sizeof(VAL)));
    return Span<const VAL>(static_cast<const VAL*>(ptr()), 1);
  }
}

template <>
inline Legion::DomainPoint Scalar::value<Legion::DomainPoint>() const
{
  Legion::DomainPoint result;
  auto span  = values<int64_t>();
  result.dim = span.size();
  for (auto idx = 0; idx < result.dim; ++idx) result[idx] = span[idx];
  return result;
}

}  // namespace legate
