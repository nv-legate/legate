/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>  // std::addressof
#include <string_view>
#include <type_traits>

// Useful for IDEs
#include <legate/data/scalar.h>

namespace legate {

namespace detail {

template <typename T>
inline decltype(auto) canonical_value_of(T&& v) noexcept
{
  return std::forward<T>(v);
}

inline std::uint64_t canonical_value_of(std::size_t v) noexcept { return std::uint64_t{v}; }

}  // namespace detail

template <typename T>
Scalar::Scalar(const T& value, private_tag)
  : Scalar{create_impl_(primitive_type(type_code_of_v<T>), std::addressof(value), /* copy */ true),
           private_tag{}}
{
  static_assert(type_code_of_v<T> != Type::Code::FIXED_ARRAY);
  static_assert(type_code_of_v<T> != Type::Code::STRUCT);
  static_assert(type_code_of_v<T> != Type::Code::STRING);
  static_assert(type_code_of_v<T> != Type::Code::NIL);
}

template <typename T, typename SFINAE>
Scalar::Scalar(const T& value) : Scalar{detail::canonical_value_of(value), private_tag{}}
{
}

template <typename T>
Scalar::Scalar(const T& value, const Type& type)
  : Scalar{checked_create_impl_(type, std::addressof(value), /* copy */ true, sizeof(T)),
           private_tag{}}
{
}

template <typename T>
Scalar::Scalar(const std::vector<T>& values)
  : Scalar{checked_create_impl_(fixed_array_type(primitive_type(type_code_of_v<T>), values.size()),
                                values.data(),
                                /* copy */ true,
                                values.size() * sizeof(T)),
           private_tag{}}
{
  static_assert(type_code_of_v<T> != Type::Code::FIXED_ARRAY);
  static_assert(type_code_of_v<T> != Type::Code::STRUCT);
  static_assert(type_code_of_v<T> != Type::Code::STRING);
  static_assert(type_code_of_v<T> != Type::Code::NIL);
}

template <typename T>
Scalar::Scalar(const tuple<T>& values) : Scalar{values.data()}
{
}

template <std::int32_t DIM>
Scalar::Scalar(const Point<DIM>& point)
  : Scalar{create_impl_(point_type(DIM), &point, /* copy */ true), private_tag{}}
{
  static_assert(DIM <= LEGATE_MAX_DIM);
}

template <std::int32_t DIM>
Scalar::Scalar(const Rect<DIM>& rect)
  : Scalar{create_impl_(rect_type(DIM), &rect, /* copy */ true), private_tag{}}
{
  static_assert(DIM <= LEGATE_MAX_DIM);
}

template <typename VAL>
VAL Scalar::value() const
{
  const auto ty = type();

  if (ty.code() == Type::Code::STRING) {
    throw_invalid_type_conversion_exception_("string", "other types");
  }
  if (sizeof(VAL) != ty.size()) {
    throw_invalid_size_exception_(ty.size(), sizeof(VAL));
  }
  return *static_cast<const VAL*>(ptr());
}

// These are defined in the .cpp
template <>
std::string_view Scalar::value() const;

template <>
std::string Scalar::value() const;

template <>
Legion::DomainPoint Scalar::value<Legion::DomainPoint>() const;

template <typename VAL>
Span<const VAL> Scalar::values() const
{
  const auto ty = type();

  switch (const auto code = ty.code()) {
    case Type::Code::FIXED_ARRAY: {
      const auto [ptr, size] = make_fixed_array_values_(sizeof(VAL));

      return Span<const VAL>(static_cast<const VAL*>(ptr), size);
    }
    case Type::Code::STRING: {
      using char_type = typename type_of_t<Type::Code::STRING>::value_type;

      if constexpr (std::is_same_v<VAL, bool>) {
        throw_invalid_type_conversion_exception_("string", "Span<bool>");
      }
      if constexpr (sizeof(VAL) != sizeof(char_type)) {
        throw_invalid_span_conversion_exception_(code, "size", sizeof(char_type), sizeof(VAL));
      }
      if constexpr (alignof(VAL) != alignof(char_type)) {
        throw_invalid_span_conversion_exception_(
          code, "alignment", alignof(char_type), alignof(VAL));
      }

      const auto [ptr, size] = make_string_values_();

      return Span<const VAL>(static_cast<const VAL*>(ptr), size);
    }
    case Type::Code::NIL: return Span<const VAL>(nullptr, 0);
    case Type::Code::BOOL: [[fallthrough]];
    case Type::Code::INT8: [[fallthrough]];
    case Type::Code::INT16: [[fallthrough]];
    case Type::Code::INT32: [[fallthrough]];
    case Type::Code::INT64: [[fallthrough]];
    case Type::Code::UINT8: [[fallthrough]];
    case Type::Code::UINT16: [[fallthrough]];
    case Type::Code::UINT32: [[fallthrough]];
    case Type::Code::UINT64: [[fallthrough]];
    case Type::Code::FLOAT16: [[fallthrough]];
    case Type::Code::FLOAT32: [[fallthrough]];
    case Type::Code::FLOAT64: [[fallthrough]];
    case Type::Code::COMPLEX64: [[fallthrough]];
    case Type::Code::COMPLEX128: [[fallthrough]];
    case Type::Code::BINARY:
      [[fallthrough]];
      // TODO(jfaibussowit)
      // STRUCT and LIST should very likely be handled differently, I cannot imagine that
      // casting a pointer to the data is sufficient to properly handle these
    case Type::Code::STRUCT: [[fallthrough]];
    case Type::Code::LIST: break;
  }

  if (sizeof(VAL) != ty.size()) {
    throw_invalid_size_exception_(ty.size(), sizeof(VAL));
  }
  return Span<const VAL>(static_cast<const VAL*>(ptr()), 1);
}

inline const SharedPtr<detail::Scalar>& Scalar::impl() { return impl_; }

inline const SharedPtr<detail::Scalar>& Scalar::impl() const { return impl_; }

}  // namespace legate
