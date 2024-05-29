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

#include <cstddef>
#include <cstdint>
#include <memory>  // std::addressof
#include <string_view>
#include <type_traits>

// Useful for IDEs
#include "core/data/scalar.h"

namespace legate {

namespace detail {

template <typename T>
constexpr legate::Type::Code canonical_type_code_of() noexcept
{
  using legate::Type;  // to disambiguate from legate::detail::Type;

  if constexpr (std::is_same_v<size_t, T>) {
    static_assert(sizeof(T) == sizeof(std::uint64_t));
    return Type::Code::UINT64;
  } else {
    constexpr auto ret = type_code_of<T>;

    static_assert(ret != Type::Code::FIXED_ARRAY);
    static_assert(ret != Type::Code::STRUCT);
    static_assert(ret != Type::Code::STRING);
    static_assert(ret != Type::Code::NIL);
    return ret;
  }
}

template <typename T>
inline decltype(auto) canonical_value_of(T&& v) noexcept
{
  return std::forward<T>(v);
}

inline std::uint64_t canonical_value_of(std::size_t v) noexcept { return std::uint64_t{v}; }

}  // namespace detail

template <typename T>
Scalar::Scalar(T value, private_tag)
  : Scalar{create_impl_(
             primitive_type(detail::canonical_type_code_of<T>()), std::addressof(value), true),
           private_tag{}}

{
}

template <typename T, typename SFINAE>
Scalar::Scalar(T value) : Scalar{detail::canonical_value_of(std::move(value)), private_tag{}}
{
}

template <typename T>
Scalar::Scalar(T value, const Type& type)
  : Scalar{checked_create_impl_(type, std::addressof(value), true, sizeof(T)), private_tag{}}
{
}

template <typename T>
Scalar::Scalar(const std::vector<T>& values)
  : Scalar{checked_create_impl_(
             fixed_array_type(primitive_type(detail::canonical_type_code_of<T>()), values.size()),
             values.data(),
             true,
             values.size() * sizeof(T)),
           private_tag{}}
{
}

template <typename T>
Scalar::Scalar(const tuple<T>& values) : Scalar{values.data()}
{
}

template <std::int32_t DIM>
Scalar::Scalar(const Point<DIM>& point)
  : Scalar{create_impl_(point_type(DIM), &point, true), private_tag{}}
{
}

template <std::int32_t DIM>
Scalar::Scalar(const Rect<DIM>& rect)
  : Scalar{create_impl_(rect_type(DIM), &rect, true), private_tag{}}
{
  static_assert(DIM <= LEGATE_MAX_DIM);
}

template <typename VAL>
VAL Scalar::value() const
{
  const auto ty = type();

  if (ty.code() == Type::Code::STRING) {
    throw std::invalid_argument{"String cannot be casted to other types"};
  }
  if (sizeof(VAL) != ty.size()) {
    throw std::invalid_argument{"Size of the scalar is " + std::to_string(ty.size()) +
                                ", but the requested type has size " + std::to_string(sizeof(VAL))};
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
  auto ty = type();
  if (ty.code() == Type::Code::FIXED_ARRAY) {
    auto arr_type  = ty.as_fixed_array_type();
    auto elem_type = arr_type.element_type();
    if (sizeof(VAL) != elem_type.size()) {
      throw std::invalid_argument{
        "The scalar's element type has size " + std::to_string(elem_type.size()) +
        ", but the requested element type has size " + std::to_string(sizeof(VAL))};
    }
    auto size = arr_type.num_elements();
    return {reinterpret_cast<const VAL*>(ptr()), size};
  }

  if (ty.code() == Type::Code::STRING) {
    if (sizeof(VAL) != 1) {
      throw std::invalid_argument{
        "String scalar can only be converted into a span of a type whose size is 1 byte"};
    }
    auto data         = ptr();
    auto len          = *static_cast<const uint32_t*>(data);
    const auto* begin = static_cast<const char*>(data) + sizeof(uint32_t);
    return {reinterpret_cast<const VAL*>(begin), len};
  }
  if (ty.code() == Type::Code::NIL) {
    return {nullptr, 0};
  }
  if (sizeof(VAL) != ty.size()) {
    throw std::invalid_argument{"Size of the scalar is " + std::to_string(ty.size()) +
                                ", but the requested element type has size " +
                                std::to_string(sizeof(VAL))};
  }
  return {static_cast<const VAL*>(ptr()), 1};
}

inline const SharedPtr<detail::Scalar>& Scalar::impl() { return impl_; }

inline const SharedPtr<detail::Scalar>& Scalar::impl() const { return impl_; }

// ==========================================================================================

inline Scalar null() { return {}; }

}  // namespace legate
