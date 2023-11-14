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

#include "core/type/type_info.h"

#include "legate_defines.h"

#include <limits.h>

#ifdef LEGION_REDOP_COMPLEX
#ifdef LEGION_REDOP_HALF
#define COMPLEX_HALF
#endif
#include "mathtypes/complex.h"
#endif

#ifdef LEGION_REDOP_HALF
#include "mathtypes/half.h"
#endif

/**
 * @file
 * @brief Definitions for type traits in Legate
 */

namespace legate {

// This maps a type to its type code
#if defined(__clang__) && !defined(__NVCC__)
#define PREFIX static
#else
#define PREFIX
#endif

/**
 * @ingroup util
 * @brief A template constexpr that converts types to type codes
 */
template <class>
PREFIX constexpr Type::Code type_code_of = Type::Code::NIL;
template <>
PREFIX constexpr Type::Code type_code_of<__half> = Type::Code::FLOAT16;
template <>
PREFIX constexpr Type::Code type_code_of<float> = Type::Code::FLOAT32;
template <>
PREFIX constexpr Type::Code type_code_of<double> = Type::Code::FLOAT64;
template <>
PREFIX constexpr Type::Code type_code_of<int8_t> = Type::Code::INT8;
template <>
PREFIX constexpr Type::Code type_code_of<int16_t> = Type::Code::INT16;
template <>
PREFIX constexpr Type::Code type_code_of<int32_t> = Type::Code::INT32;
template <>
PREFIX constexpr Type::Code type_code_of<int64_t> = Type::Code::INT64;
template <>
PREFIX constexpr Type::Code type_code_of<uint8_t> = Type::Code::UINT8;
template <>
PREFIX constexpr Type::Code type_code_of<uint16_t> = Type::Code::UINT16;
template <>
PREFIX constexpr Type::Code type_code_of<uint32_t> = Type::Code::UINT32;
template <>
PREFIX constexpr Type::Code type_code_of<uint64_t> = Type::Code::UINT64;
template <>
PREFIX constexpr Type::Code type_code_of<bool> = Type::Code::BOOL;
template <>
PREFIX constexpr Type::Code type_code_of<complex<float>> = Type::Code::COMPLEX64;
template <>
PREFIX constexpr Type::Code type_code_of<complex<double>> = Type::Code::COMPLEX128;
// When the CUDA build is off, complex<T> is an alias to std::complex<T>
#if LegateDefined(LEGATE_USE_CUDA)
template <>
PREFIX constexpr Type::Code type_code_of<std::complex<float>> = Type::Code::COMPLEX64;
template <>
PREFIX constexpr Type::Code type_code_of<std::complex<double>> = Type::Code::COMPLEX128;
#endif
template <>
PREFIX constexpr Type::Code type_code_of<std::string> = Type::Code::STRING;

#undef PREFIX

template <Type::Code CODE>
struct TypeOf {
  using type = void;
};
template <>
struct TypeOf<Type::Code::BOOL> {
  using type = bool;
};
template <>
struct TypeOf<Type::Code::INT8> {
  using type = int8_t;
};
template <>
struct TypeOf<Type::Code::INT16> {
  using type = int16_t;
};
template <>
struct TypeOf<Type::Code::INT32> {
  using type = int32_t;
};
template <>
struct TypeOf<Type::Code::INT64> {
  using type = int64_t;
};
template <>
struct TypeOf<Type::Code::UINT8> {
  using type = uint8_t;
};
template <>
struct TypeOf<Type::Code::UINT16> {
  using type = uint16_t;
};
template <>
struct TypeOf<Type::Code::UINT32> {
  using type = uint32_t;
};
template <>
struct TypeOf<Type::Code::UINT64> {
  using type = uint64_t;
};
template <>
struct TypeOf<Type::Code::FLOAT16> {
  using type = __half;
};
template <>
struct TypeOf<Type::Code::FLOAT32> {
  using type = float;
};
template <>
struct TypeOf<Type::Code::FLOAT64> {
  using type = double;
};
template <>
struct TypeOf<Type::Code::COMPLEX64> {
  using type = complex<float>;
};
template <>
struct TypeOf<Type::Code::COMPLEX128> {
  using type = complex<double>;
};
template <>
struct TypeOf<Type::Code::STRING> {
  using type = std::string;
};

/**
 * @ingroup util
 * @brief A template that converts type codes to types
 */
template <Type::Code CODE>
using type_of = typename TypeOf<CODE>::type;

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of an integral type
 */
template <Type::Code CODE>
struct is_integral {
  static constexpr bool value = std::is_integral_v<type_of<CODE>>;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of a signed integral type
 */
template <Type::Code CODE>
struct is_signed {
  static constexpr bool value = std::is_signed_v<type_of<CODE>>;
};
template <>
struct is_signed<Type::Code::FLOAT16> {
  static constexpr bool value = true;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of an unsigned integral type
 */
template <Type::Code CODE>
struct is_unsigned {
  static constexpr bool value = std::is_unsigned_v<type_of<CODE>>;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of a floating point type
 */
template <Type::Code CODE>
struct is_floating_point {
  static constexpr bool value = std::is_floating_point_v<type_of<CODE>>;
};

template <>
struct is_floating_point<Type::Code::FLOAT16> {
  static constexpr bool value = true;
};

/**
 * @ingroup util
 * @brief A predicate that holds if the type code is of a complex type
 */
template <Type::Code CODE>
struct is_complex : std::false_type {};

template <>
struct is_complex<Type::Code::COMPLEX64> : std::true_type {};

template <>
struct is_complex<Type::Code::COMPLEX128> : std::true_type {};

/**
 * @ingroup util
 * @brief A predicate that holds if the type is one of the supported complex types
 */
template <typename T>
struct is_complex_type : std::false_type {};

template <>
struct is_complex_type<complex<float>> : std::true_type {};

template <>
struct is_complex_type<complex<double>> : std::true_type {};

// When the CUDA build is off, complex<T> is an alias to std::complex<T>
#if LegateDefined(LEGATE_USE_CUDA)
template <>
struct is_complex_type<std::complex<float>> : std::true_type {};

template <>
struct is_complex_type<std::complex<double>> : std::true_type {};
#endif

}  // namespace legate
