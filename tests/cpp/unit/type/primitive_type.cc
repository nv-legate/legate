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

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace primitive_type_test {

namespace {

using PrimitiveTypeUnit = DefaultFixture;

class PrimitiveTypeTest
  : public PrimitiveTypeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, legate::Type::Code>> {};

class PrimitiveTypeFeatureTest
  : public PrimitiveTypeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type,
                                                    legate::Type::Code,
                                                    std::string,
                                                    std::uint32_t /* size */,
                                                    std::uint32_t /* alignment */>> {};

class NegativeTypeTest : public PrimitiveTypeUnit,
                         public ::testing::WithParamInterface<legate::Type::Code> {};

INSTANTIATE_TEST_SUITE_P(
  PrimitiveTypeUnit,
  PrimitiveTypeTest,
  ::testing::Values(std::make_tuple(legate::null_type(), legate::Type::Code::NIL),
                    std::make_tuple(legate::bool_(), legate::Type::Code::BOOL),
                    std::make_tuple(legate::int8(), legate::Type::Code::INT8),
                    std::make_tuple(legate::int16(), legate::Type::Code::INT16),
                    std::make_tuple(legate::int32(), legate::Type::Code::INT32),
                    std::make_tuple(legate::int64(), legate::Type::Code::INT64),
                    std::make_tuple(legate::uint8(), legate::Type::Code::UINT8),
                    std::make_tuple(legate::uint16(), legate::Type::Code::UINT16),
                    std::make_tuple(legate::uint32(), legate::Type::Code::UINT32),
                    std::make_tuple(legate::uint64(), legate::Type::Code::UINT64),
                    std::make_tuple(legate::float16(), legate::Type::Code::FLOAT16),
                    std::make_tuple(legate::float32(), legate::Type::Code::FLOAT32),
                    std::make_tuple(legate::float64(), legate::Type::Code::FLOAT64),
                    std::make_tuple(legate::complex64(), legate::Type::Code::COMPLEX64),
                    std::make_tuple(legate::complex128(), legate::Type::Code::COMPLEX128)));

INSTANTIATE_TEST_SUITE_P(
  PrimitiveTypeUnit,
  PrimitiveTypeFeatureTest,
  ::testing::Values(
    std::make_tuple(legate::null_type(), legate::Type::Code::NIL, "null_type", 0, 0),
    std::make_tuple(legate::bool_(), legate::Type::Code::BOOL, "bool", sizeof(bool), alignof(bool)),
    std::make_tuple(
      legate::int8(), legate::Type::Code::INT8, "int8", sizeof(std::int8_t), alignof(std::int8_t)),
    std::make_tuple(legate::int16(),
                    legate::Type::Code::INT16,
                    "int16",
                    sizeof(std::int16_t),
                    alignof(std::int16_t)),
    std::make_tuple(legate::int32(),
                    legate::Type::Code::INT32,
                    "int32",
                    sizeof(std::int32_t),
                    alignof(std::int32_t)),
    std::make_tuple(legate::int64(),
                    legate::Type::Code::INT64,
                    "int64",
                    sizeof(std::int64_t),
                    alignof(std::int64_t)),
    std::make_tuple(legate::uint8(),
                    legate::Type::Code::UINT8,
                    "uint8",
                    sizeof(std::uint8_t),
                    alignof(std::uint8_t)),
    std::make_tuple(legate::uint16(),
                    legate::Type::Code::UINT16,
                    "uint16",
                    sizeof(std::uint16_t),
                    alignof(std::uint16_t)),
    std::make_tuple(legate::uint32(),
                    legate::Type::Code::UINT32,
                    "uint32",
                    sizeof(std::uint32_t),
                    alignof(std::uint32_t)),
    std::make_tuple(legate::uint64(),
                    legate::Type::Code::UINT64,
                    "uint64",
                    sizeof(std::uint64_t),
                    alignof(std::uint64_t)),
    std::make_tuple(
      legate::float16(), legate::Type::Code::FLOAT16, "float16", sizeof(__half), alignof(__half)),
    std::make_tuple(
      legate::float32(), legate::Type::Code::FLOAT32, "float32", sizeof(float), alignof(float)),
    std::make_tuple(
      legate::float64(), legate::Type::Code::FLOAT64, "float64", sizeof(double), alignof(double)),
    std::make_tuple(legate::complex64(),
                    legate::Type::Code::COMPLEX64,
                    "complex64",
                    sizeof(complex<float>),
                    alignof(complex<float>)),
    std::make_tuple(legate::complex128(),
                    legate::Type::Code::COMPLEX128,
                    "complex128",
                    sizeof(complex<double>),
                    alignof(complex<double>))));

INSTANTIATE_TEST_SUITE_P(PrimitiveTypeUnit,
                         NegativeTypeTest,
                         ::testing::Values(legate::Type::Code::FIXED_ARRAY,
                                           legate::Type::Code::STRUCT,
                                           legate::Type::Code::STRING,
                                           legate::Type::Code::LIST,
                                           legate::Type::Code::BINARY));

}  // namespace

TEST_P(PrimitiveTypeFeatureTest, Basic)
{
  const auto [type, code, to_string, size, alignment] = GetParam();

  ASSERT_EQ(type.code(), code);
  ASSERT_EQ(type.size(), size);
  ASSERT_EQ(type.alignment(), alignment);
  ASSERT_FALSE(type.variable_size());
  ASSERT_TRUE(type.is_primitive());
  ASSERT_EQ(type.to_string(), to_string);

  // Note: aim to test the copy initialization of Type
  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

TEST_P(PrimitiveTypeTest, Basic)
{
  const auto [type, code] = GetParam();

  ASSERT_EQ(type, legate::primitive_type(code));
}

TEST_P(NegativeTypeTest, PrimitiveType)
{
  ASSERT_THROW(static_cast<void>(legate::primitive_type(GetParam())), std::invalid_argument);
}

}  // namespace primitive_type_test
