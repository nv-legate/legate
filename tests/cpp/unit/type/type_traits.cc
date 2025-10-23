/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/type_traits.h>

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
static_assert(legate::type_code_of_v<legate::Half> == legate::Type::Code::FLOAT16);
static_assert(legate::type_code_of_v<float> == legate::Type::Code::FLOAT32);
static_assert(legate::type_code_of_v<double> == legate::Type::Code::FLOAT64);
static_assert(legate::type_code_of_v<legate::Complex<float>> == legate::Type::Code::COMPLEX64);
static_assert(legate::type_code_of_v<legate::Complex<double>> == legate::Type::Code::COMPLEX128);
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
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FLOAT16>, legate::Half>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FLOAT32>, float>);
static_assert(std::is_same_v<legate::type_of_t<legate::Type::Code::FLOAT64>, double>);
static_assert(
  std::is_same_v<legate::type_of_t<legate::Type::Code::COMPLEX64>, legate::Complex<float>>);
static_assert(
  std::is_same_v<legate::type_of_t<legate::Type::Code::COMPLEX128>, legate::Complex<double>>);
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
static_assert(legate::is_complex_type<legate::Complex<float>>::value);
static_assert(legate::is_complex_type<legate::Complex<double>>::value);

static_assert(!legate::is_complex_type<bool>::value);
static_assert(!legate::is_complex_type<std::int8_t>::value);
static_assert(!legate::is_complex_type<std::int16_t>::value);
static_assert(!legate::is_complex_type<std::int32_t>::value);
static_assert(!legate::is_complex_type<std::int64_t>::value);
static_assert(!legate::is_complex_type<std::uint8_t>::value);
static_assert(!legate::is_complex_type<std::uint16_t>::value);
static_assert(!legate::is_complex_type<std::uint32_t>::value);
static_assert(!legate::is_complex_type<std::uint64_t>::value);
static_assert(!legate::is_complex_type<legate::Half>::value);
static_assert(!legate::is_complex_type<float>::value);
static_assert(!legate::is_complex_type<double>::value);

static_assert(!legate::is_complex_type<void>::value);
static_assert(!legate::is_complex_type<std::string>::value);

}  // namespace type_traits_test

#include <cuda/std/functional>

#include <vector>

namespace is_instance_of_test {

template <typename T>
class TemplateClass {};

template <typename T = void>
class TemplateClassWithDefault {};

class NotTemplate {};

static_assert(legate::detail::is_instance_of_v<TemplateClass<int>, TemplateClass>);
static_assert(!legate::detail::is_instance_of_v<int, TemplateClass>);

static_assert(
  legate::detail::is_instance_of_v<TemplateClassWithDefault<>, TemplateClassWithDefault>);
static_assert(
  legate::detail::is_instance_of_v<TemplateClassWithDefault<void>, TemplateClassWithDefault>);
static_assert(
  legate::detail::is_instance_of_v<TemplateClassWithDefault<int>, TemplateClassWithDefault>);
static_assert(!legate::detail::is_instance_of_v<void, TemplateClassWithDefault>);
static_assert(!legate::detail::is_instance_of_v<NotTemplate, TemplateClass>);

// Test some "real world" types
static_assert(legate::detail::is_instance_of_v<std::vector<int>, std::vector>);
static_assert(legate::detail::is_instance_of_v<cuda::std::plus<>, cuda::std::plus>);
static_assert(legate::detail::is_instance_of_v<cuda::std::plus<int>, cuda::std::plus>);
static_assert(legate::detail::is_instance_of_v<cuda::std::minus<>, cuda::std::minus>);
static_assert(legate::detail::is_instance_of_v<cuda::std::minus<int>, cuda::std::minus>);
static_assert(legate::detail::is_instance_of_v<cuda::std::bit_and<>, cuda::std::bit_and>);
static_assert(legate::detail::is_instance_of_v<cuda::std::bit_and<int>, cuda::std::bit_and>);

static_assert(legate::detail::is_instance_of_v<cuda::std::bit_or<>, cuda::std::bit_or>);
static_assert(legate::detail::is_instance_of_v<cuda::std::bit_or<int>, cuda::std::bit_or>);
static_assert(legate::detail::is_instance_of_v<cuda::std::bit_xor<>, cuda::std::bit_xor>);
static_assert(legate::detail::is_instance_of_v<cuda::std::bit_xor<int>, cuda::std::bit_xor>);

}  // namespace is_instance_of_test
