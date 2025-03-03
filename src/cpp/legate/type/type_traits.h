/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/macros.h>

#include <climits>
#include <cstdint>
#include <string>
#include <type_traits>

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <complex>
#endif

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
#define COMPLEX_HALF
#endif
#include <mathtypes/complex.h>
#endif

#ifdef LEGION_REDOP_HALF
#include <mathtypes/half.h>
#endif

/**
 * @file
 * @brief Definitions for type traits in Legate
 */

namespace legate {

/**
 * @addtogroup type
 * @{
 */

// TODO(jfaibussowit)
// Once the deprecation period elapses for type_code_of, we should move this to the primary
// namespace
namespace type_code_of_detail {

template <typename T, typename SFINAE = void>
struct type_code_of  // NOLINT(readability-identifier-naming)
  : std::integral_constant<Type::Code, Type::Code::NIL> {};

template <>
struct type_code_of<__half> : std::integral_constant<Type::Code, Type::Code::FLOAT16> {};
template <>
struct type_code_of<float> : std::integral_constant<Type::Code, Type::Code::FLOAT32> {};
template <>
struct type_code_of<double> : std::integral_constant<Type::Code, Type::Code::FLOAT64> {};
template <>
struct type_code_of<std::int8_t> : std::integral_constant<Type::Code, Type::Code::INT8> {};
template <>
struct type_code_of<std::int16_t> : std::integral_constant<Type::Code, Type::Code::INT16> {};
template <>
struct type_code_of<std::int32_t> : std::integral_constant<Type::Code, Type::Code::INT32> {};
template <>
struct type_code_of<std::int64_t> : std::integral_constant<Type::Code, Type::Code::INT64> {};
template <>
struct type_code_of<std::uint8_t> : std::integral_constant<Type::Code, Type::Code::UINT8> {};
template <>
struct type_code_of<std::uint16_t> : std::integral_constant<Type::Code, Type::Code::UINT16> {};
template <>
struct type_code_of<std::uint32_t> : std::integral_constant<Type::Code, Type::Code::UINT32> {};
template <>
struct type_code_of<std::uint64_t> : std::integral_constant<Type::Code, Type::Code::UINT64> {};
// Do not be fooled by the template parameter. This matches *exactly* std::size_t if and only
// if it is not the same as std::uint64_t. It needs to be a template because otherwise you
// cannot use std::enable_if_t.
//
// This specialization is needed because on some systems (e.g. macOS) std::size_t !=
// std::uint64_t.
template <typename T>
struct type_code_of<
  T,
  std::enable_if_t<std::is_same_v<T, std::size_t> && !std::is_same_v<std::size_t, std::uint64_t>>>
  : type_code_of<std::uint64_t> {
  static_assert(sizeof(T) == sizeof(std::uint64_t));
  static_assert(alignof(T) == alignof(std::uint64_t));
};
template <>
struct type_code_of<bool> : std::integral_constant<Type::Code, Type::Code::BOOL> {};
template <>
struct type_code_of<std::string> : std::integral_constant<Type::Code, Type::Code::STRING> {};
template <>
struct type_code_of<complex<float>> : std::integral_constant<Type::Code, Type::Code::COMPLEX64> {};
template <>
struct type_code_of<complex<double>> : std::integral_constant<Type::Code, Type::Code::COMPLEX128> {
};
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
template <>
struct type_code_of<std::complex<float>>
  : std::integral_constant<Type::Code, Type::Code::COMPLEX64> {};
template <>
struct type_code_of<std::complex<double>>
  : std::integral_constant<Type::Code, Type::Code::COMPLEX128> {};
#endif

template <typename T>
struct type_code_of<T*> : std::integral_constant<Type::Code, Type::Code::UINT64> {
  static_assert(sizeof(T*) == sizeof(std::uint64_t));
  static_assert(alignof(T*) == alignof(std::uint64_t));
};

template <typename T>
struct type_code_of<T, std::enable_if_t<std::is_enum_v<T>>>
  : type_code_of<std::underlying_type_t<T>> {};

}  // namespace type_code_of_detail

/**
 * @brief A template constexpr that converts types to type codes
 */
template <typename T>
inline constexpr Type::Code type_code_of_v = type_code_of_detail::type_code_of<T>::value;

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
inline constexpr Type::Code type_code_of [[deprecated("use legate::type_code_of_v instead")]] =
  type_code_of_v<T>;
// NOLINTEND(readability-identifier-naming)

// TODO(jfaibussowit)
// Move this to top-level namespace once deprecation period elapses.
namespace type_of_detail {

template <Type::Code CODE>
struct type_of {  // NOLINT(readability-identifier-naming)
  using type = void;
};
template <>
struct type_of<Type::Code::BOOL> {
  using type = bool;
};
template <>
struct type_of<Type::Code::INT8> {
  using type = std::int8_t;
};
template <>
struct type_of<Type::Code::INT16> {
  using type = std::int16_t;
};
template <>
struct type_of<Type::Code::INT32> {
  using type = std::int32_t;
};
template <>
struct type_of<Type::Code::INT64> {
  using type = std::int64_t;
};
template <>
struct type_of<Type::Code::UINT8> {
  using type = std::uint8_t;
};
template <>
struct type_of<Type::Code::UINT16> {
  using type = std::uint16_t;
};
template <>
struct type_of<Type::Code::UINT32> {
  using type = std::uint32_t;
};
template <>
struct type_of<Type::Code::UINT64> {
  using type = std::uint64_t;
};
template <>
struct type_of<Type::Code::FLOAT16> {
  using type = __half;
};
template <>
struct type_of<Type::Code::FLOAT32> {
  using type = float;
};
template <>
struct type_of<Type::Code::FLOAT64> {
  using type = double;
};
template <>
struct type_of<Type::Code::COMPLEX64> {
  using type = complex<float>;
};
template <>
struct type_of<Type::Code::COMPLEX128> {
  using type = complex<double>;
};
template <>
struct type_of<Type::Code::STRING> {
  using type = std::string;
};

}  // namespace type_of_detail

/**
 * @brief A template that converts type codes to types
 */
template <Type::Code CODE>
using type_of_t = typename type_of_detail::type_of<CODE>::type;

template <Type::Code CODE>
using type_of [[deprecated("use legate::type_of_t instead")]] = type_of_t<CODE>;

/**
 * @brief A predicate that holds if the type code is of an integral type
 */
template <Type::Code CODE>
struct is_integral : std::is_integral<type_of_t<CODE>> {};

/**
 * @brief A predicate that holds if the type code is of a signed integral type
 */
template <Type::Code CODE>
struct is_signed : std::is_signed<type_of_t<CODE>> {};

template <>
struct is_signed<Type::Code::FLOAT16> : std::true_type {};

/**
 * @brief A predicate that holds if the type code is of an unsigned integral type
 */
template <Type::Code CODE>
struct is_unsigned : std::is_unsigned<type_of_t<CODE>> {};

/**
 * @brief A predicate that holds if the type code is of a floating point type
 */
template <Type::Code CODE>
struct is_floating_point : std::is_floating_point<type_of_t<CODE>> {};

template <>
struct is_floating_point<Type::Code::FLOAT16> : std::true_type {};

/**
 * @brief A predicate that holds if the type code is of a complex type
 */
template <Type::Code CODE>
struct is_complex : std::false_type {};

template <>
struct is_complex<Type::Code::COMPLEX64> : std::true_type {};

template <>
struct is_complex<Type::Code::COMPLEX128> : std::true_type {};

/**
 * @brief A predicate that holds if the type is one of the supported complex types
 */
template <typename T>
struct is_complex_type : std::false_type {};

template <>
struct is_complex_type<complex<float>> : std::true_type {};

template <>
struct is_complex_type<complex<double>> : std::true_type {};

// When the CUDA build is off, complex<T> is an alias to std::complex<T>
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
template <>
struct is_complex_type<std::complex<float>> : std::true_type {};

template <>
struct is_complex_type<std::complex<double>> : std::true_type {};
#endif

/** @} */

}  // namespace legate
