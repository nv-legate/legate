/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

namespace type_traits_test {

static_assert(legate::type_code_of_v<void> == legate::Type::Code::NIL);
static_assert(legate::type_code_of_v<bool> == legate::Type::Code::BOOL);
static_assert(legate::type_code_of_v<std::int8_t> == legate::Type::Code::INT8);
static_assert(legate::type_code_of_v<std::int16_t> == legate::Type::Code::INT16);
static_assert(legate::type_code_of_v<std::int32_t> == legate::Type::Code::INT32);
static_assert(legate::type_code_of_v<std::int64_t> == legate::Type::Code::INT64);
static_assert(legate::type_code_of_v<std::uint8_t> == legate::Type::Code::UINT8);
static_assert(legate::type_code_of_v<std::uint16_t> == legate::Type::Code::UINT16);
static_assert(legate::type_code_of_v<std::uint32_t> == legate::Type::Code::UINT32);
static_assert(legate::type_code_of_v<std::uint64_t> == legate::Type::Code::UINT64);
static_assert(legate::type_code_of_v<__half> == legate::Type::Code::FLOAT16);
static_assert(legate::type_code_of_v<float> == legate::Type::Code::FLOAT32);
static_assert(legate::type_code_of_v<double> == legate::Type::Code::FLOAT64);
static_assert(legate::type_code_of_v<complex<float>> == legate::Type::Code::COMPLEX64);
static_assert(legate::type_code_of_v<complex<double>> == legate::Type::Code::COMPLEX128);
static_assert(legate::type_code_of_v<std::string> == legate::Type::Code::STRING);

static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::BOOL>, bool>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::INT8>, std::int8_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::INT16>, std::int16_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::INT32>, std::int32_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::INT64>, std::int64_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::UINT8>, std::uint8_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::UINT16>, std::uint16_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::UINT32>, std::uint32_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::UINT64>, std::uint64_t>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FLOAT16>, __half>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FLOAT32>, float>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FLOAT64>, double>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::COMPLEX64>, complex<float>>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::COMPLEX128>, complex<double>>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::STRING>, std::string>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::NIL>, void>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FIXED_ARRAY>, void>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::STRUCT>, void>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::LIST>, void>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::BINARY>, void>);

// is_integral
static_assert(legate::is_integral<legate::Type::Code::BOOL>::value);
static_assert(legate::is_integral<legate::Type::Code::INT8>::value);
static_assert(legate::is_integral<legate::Type::Code::INT16>::value);
static_assert(legate::is_integral<legate::Type::Code::INT32>::value);
static_assert(legate::is_integral<legate::Type::Code::INT64>::value);
static_assert(legate::is_integral<legate::Type::Code::UINT8>::value);
static_assert(legate::is_integral<legate::Type::Code::UINT16>::value);
static_assert(legate::is_integral<legate::Type::Code::UINT32>::value);
static_assert(legate::is_integral<legate::Type::Code::UINT64>::value);

static_assert(!legate::is_integral<legate::Type::Code::FLOAT16>::value);
static_assert(!legate::is_integral<legate::Type::Code::FLOAT32>::value);
static_assert(!legate::is_integral<legate::Type::Code::FLOAT64>::value);
static_assert(!legate::is_integral<legate::Type::Code::COMPLEX64>::value);
static_assert(!legate::is_integral<legate::Type::Code::COMPLEX128>::value);

static_assert(!legate::is_integral<legate::Type::Code::NIL>::value);
static_assert(!legate::is_integral<legate::Type::Code::STRING>::value);
static_assert(!legate::is_integral<legate::Type::Code::FIXED_ARRAY>::value);
static_assert(!legate::is_integral<legate::Type::Code::STRUCT>::value);
static_assert(!legate::is_integral<legate::Type::Code::LIST>::value);
static_assert(!legate::is_integral<legate::Type::Code::BINARY>::value);

// is_signed
static_assert(legate::is_signed<legate::Type::Code::INT8>::value);
static_assert(legate::is_signed<legate::Type::Code::INT16>::value);
static_assert(legate::is_signed<legate::Type::Code::INT32>::value);
static_assert(legate::is_signed<legate::Type::Code::INT64>::value);
static_assert(legate::is_signed<legate::Type::Code::FLOAT32>::value);
static_assert(legate::is_signed<legate::Type::Code::FLOAT64>::value);
static_assert(legate::is_signed<legate::Type::Code::FLOAT16>::value);

static_assert(!legate::is_signed<legate::Type::Code::BOOL>::value);
static_assert(!legate::is_signed<legate::Type::Code::UINT8>::value);
static_assert(!legate::is_signed<legate::Type::Code::UINT16>::value);
static_assert(!legate::is_signed<legate::Type::Code::UINT32>::value);
static_assert(!legate::is_signed<legate::Type::Code::UINT64>::value);
static_assert(!legate::is_signed<legate::Type::Code::COMPLEX64>::value);
static_assert(!legate::is_signed<legate::Type::Code::COMPLEX128>::value);

static_assert(!legate::is_signed<legate::Type::Code::NIL>::value);
static_assert(!legate::is_signed<legate::Type::Code::STRING>::value);
static_assert(!legate::is_signed<legate::Type::Code::FIXED_ARRAY>::value);
static_assert(!legate::is_signed<legate::Type::Code::STRUCT>::value);
static_assert(!legate::is_signed<legate::Type::Code::LIST>::value);
static_assert(!legate::is_signed<legate::Type::Code::BINARY>::value);

// is_unsigned
static_assert(legate::is_unsigned<legate::Type::Code::BOOL>::value);
static_assert(legate::is_unsigned<legate::Type::Code::UINT8>::value);
static_assert(legate::is_unsigned<legate::Type::Code::UINT16>::value);
static_assert(legate::is_unsigned<legate::Type::Code::UINT32>::value);
static_assert(legate::is_unsigned<legate::Type::Code::UINT64>::value);

static_assert(!legate::is_unsigned<legate::Type::Code::INT8>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::INT16>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::INT32>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::INT64>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::FLOAT16>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::FLOAT32>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::FLOAT64>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::COMPLEX64>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::COMPLEX128>::value);

static_assert(!legate::is_unsigned<legate::Type::Code::NIL>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::STRING>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::FIXED_ARRAY>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::STRUCT>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::LIST>::value);
static_assert(!legate::is_unsigned<legate::Type::Code::BINARY>::value);

// is_floating_point
static_assert(legate::is_floating_point<legate::Type::Code::FLOAT16>::value);
static_assert(legate::is_floating_point<legate::Type::Code::FLOAT32>::value);
static_assert(legate::is_floating_point<legate::Type::Code::FLOAT64>::value);

static_assert(!legate::is_floating_point<legate::Type::Code::BOOL>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::UINT8>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::UINT16>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::UINT32>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::UINT64>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::INT8>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::INT16>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::INT32>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::INT64>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::COMPLEX64>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::COMPLEX128>::value);

static_assert(!legate::is_floating_point<legate::Type::Code::NIL>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::STRING>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::FIXED_ARRAY>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::STRUCT>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::LIST>::value);
static_assert(!legate::is_floating_point<legate::Type::Code::BINARY>::value);

// is_complex
static_assert(legate::is_complex<legate::Type::Code::COMPLEX64>::value);
static_assert(legate::is_complex<legate::Type::Code::COMPLEX128>::value);

static_assert(!legate::is_complex<legate::Type::Code::BOOL>::value);
static_assert(!legate::is_complex<legate::Type::Code::UINT8>::value);
static_assert(!legate::is_complex<legate::Type::Code::UINT16>::value);
static_assert(!legate::is_complex<legate::Type::Code::UINT32>::value);
static_assert(!legate::is_complex<legate::Type::Code::UINT64>::value);
static_assert(!legate::is_complex<legate::Type::Code::INT8>::value);
static_assert(!legate::is_complex<legate::Type::Code::INT16>::value);
static_assert(!legate::is_complex<legate::Type::Code::INT32>::value);
static_assert(!legate::is_complex<legate::Type::Code::INT64>::value);
static_assert(!legate::is_complex<legate::Type::Code::FLOAT16>::value);
static_assert(!legate::is_complex<legate::Type::Code::FLOAT32>::value);
static_assert(!legate::is_complex<legate::Type::Code::FLOAT64>::value);

static_assert(!legate::is_complex<legate::Type::Code::NIL>::value);
static_assert(!legate::is_complex<legate::Type::Code::STRING>::value);
static_assert(!legate::is_complex<legate::Type::Code::FIXED_ARRAY>::value);
static_assert(!legate::is_complex<legate::Type::Code::STRUCT>::value);
static_assert(!legate::is_complex<legate::Type::Code::LIST>::value);
static_assert(!legate::is_complex<legate::Type::Code::BINARY>::value);

// is_complex_type
static_assert(legate::is_complex_type<complex<float>>::value);
static_assert(legate::is_complex_type<complex<double>>::value);

static_assert(!legate::is_complex_type<bool>::value);
static_assert(!legate::is_complex_type<std::int8_t>::value);
static_assert(!legate::is_complex_type<std::int16_t>::value);
static_assert(!legate::is_complex_type<std::int32_t>::value);
static_assert(!legate::is_complex_type<std::int64_t>::value);
static_assert(!legate::is_complex_type<std::uint8_t>::value);
static_assert(!legate::is_complex_type<std::uint16_t>::value);
static_assert(!legate::is_complex_type<std::uint32_t>::value);
static_assert(!legate::is_complex_type<std::uint64_t>::value);
static_assert(!legate::is_complex_type<__half>::value);
static_assert(!legate::is_complex_type<float>::value);
static_assert(!legate::is_complex_type<double>::value);

static_assert(!legate::is_complex_type<void>::value);
static_assert(!legate::is_complex_type<std::string>::value);

}  // namespace type_traits_test
