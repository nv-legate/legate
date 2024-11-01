/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate.h"
#include "utilities/utilities.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

namespace type_test {

namespace {

using TypeUnit = DefaultFixture;

class PrimitiveTypeTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type, legate::Type::Code>> {};

class PrimitiveTypeFeatureTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type,
                                                    legate::Type::Code,
                                                    std::string,
                                                    std::uint32_t /* size */,
                                                    std::uint32_t /* alignment */>> {};

class BinaryTypeTest : public DefaultFixture,
                       public ::testing::WithParamInterface<std::uint32_t> {};

class FixedArrayTypeTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t, std::string>> {};

class StructTypeTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type,
                                                    std::vector<legate::Type>,
                                                    bool,
                                                    std::uint32_t /* total_size */,
                                                    std::uint32_t /* alignment */,
                                                    std::string>> {};

class DimTest : public DefaultFixture, public ::testing::WithParamInterface<std::uint32_t> {};

class TypeTest : public DefaultFixture, public ::testing::WithParamInterface<legate::Type> {};

class PointTypeTest : public DimTest {};

class RectTypeTest : public DimTest {};

class ListTypeTest : public DefaultFixture,
                     public ::testing::WithParamInterface<std::tuple<legate::Type, std::string>> {};

class UidFixedArrayTypeTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t>> {};

class UidTest : public TypeTest {};

class ReductionOperatorTest
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type, legate::ReductionOpKind>> {};

class NegativeTypeTest : public DefaultFixture,
                         public ::testing::WithParamInterface<legate::Type::Code> {};

class NegativeDimTest : public DimTest {};

INSTANTIATE_TEST_SUITE_P(
  TypeUnit,
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
  TypeUnit,
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

INSTANTIATE_TEST_SUITE_P(TypeUnit,
                         BinaryTypeTest,
                         ::testing::Values(123, 45, 0, 0xFFFFF),
                         ::testing::PrintToStringParamName{});

INSTANTIATE_TEST_SUITE_P(
  TypeUnit,
  FixedArrayTypeTest,
  ::testing::Values(
    std::make_tuple(legate::uint64(), 20, "uint64[20]") /* element type is a primitive type */,
    std::make_tuple(legate::fixed_array_type(legate::uint16(), 10),
                    10,
                    "uint16[10][10]") /* element type is not a primitive type */,
    std::make_tuple(legate::float64(), 256, "float64[256]") /* // N > 0xFFU */));

INSTANTIATE_TEST_SUITE_P(
  TypeUnit,
  StructTypeTest,
  ::testing::Values(
    std::make_tuple(
      legate::struct_type(true, legate::int16(), legate::bool_(), legate::float64()),
      std::vector<legate::Type>({legate::int16(), legate::bool_(), legate::float64()}),
      true,
      16,
      sizeof(double),
      "{int16:0,bool:2,float64:8}"),
    std::make_tuple(
      legate::struct_type(
        std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()}), true),
      std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()}),
      true,
      24,
      sizeof(double),
      "{bool:0,float64:8,int16:16}"),
    std::make_tuple(
      legate::struct_type(false, legate::int16(), legate::bool_(), legate::float64()),
      std::vector<legate::Type>({legate::int16(), legate::bool_(), legate::float64()}),
      false,
      sizeof(int16_t) + sizeof(bool) + sizeof(double),
      sizeof(bool),
      "{int16:0,bool:2,float64:3}"),
    std::make_tuple(
      legate::struct_type(
        std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()})),
      std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()}),
      false,
      sizeof(int16_t) + sizeof(bool) + sizeof(double),
      sizeof(bool),
      "{bool:0,float64:1,int16:9}")));

INSTANTIATE_TEST_SUITE_P(TypeUnit, PointTypeTest, ::testing::Values(1, 2, 3, LEGATE_MAX_DIM));

INSTANTIATE_TEST_SUITE_P(TypeUnit, RectTypeTest, ::testing::Values(1, 2, 3, LEGATE_MAX_DIM));

INSTANTIATE_TEST_SUITE_P(
  TypeUnit,
  ListTypeTest,
  ::testing::Values(std::make_tuple(legate::bool_(), "list(bool)"),
                    std::make_tuple(legate::int8(), "list(int8)"),
                    std::make_tuple(legate::int16(), "list(int16)"),
                    std::make_tuple(legate::int32(), "list(int32)"),
                    std::make_tuple(legate::int64(), "list(int64)"),
                    std::make_tuple(legate::uint8(), "list(uint8)"),
                    std::make_tuple(legate::uint16(), "list(uint16)"),
                    std::make_tuple(legate::uint32(), "list(uint32)"),
                    std::make_tuple(legate::uint64(), "list(uint64)"),
                    std::make_tuple(legate::float16(), "list(float16)"),
                    std::make_tuple(legate::float32(), "list(float32)"),
                    std::make_tuple(legate::float64(), "list(float64)"),
                    std::make_tuple(legate::complex64(), "list(complex64)"),
                    std::make_tuple(legate::complex128(), "list(complex128)"),
                    std::make_tuple(legate::struct_type(true, legate::bool_(), legate::int32()),
                                    "list({bool:0,int32:4})"),
                    std::make_tuple(legate::struct_type(false, legate::bool_(), legate::int32()),
                                    "list({bool:0,int32:1})"),
                    std::make_tuple(legate::point_type(1), "list(int64[1])"),
                    std::make_tuple(legate::rect_type(2), "list({int64[2]:0,int64[2]:16})")));

INSTANTIATE_TEST_SUITE_P(TypeUnit,
                         UidFixedArrayTypeTest,
                         ::testing::Combine(::testing::Values(legate::bool_(),
                                                              legate::int8(),
                                                              legate::int16(),
                                                              legate::int32(),
                                                              legate::int64(),
                                                              legate::uint8(),
                                                              legate::uint16(),
                                                              legate::uint32(),
                                                              legate::uint64(),
                                                              legate::float16(),
                                                              legate::float32(),
                                                              legate::float64(),
                                                              legate::complex64(),
                                                              legate::complex128()),
                                            ::testing::Values(1, 2, 3, LEGATE_MAX_DIM)));

INSTANTIATE_TEST_SUITE_P(TypeUnit,
                         UidTest,
                         ::testing::Values(legate::bool_(),
                                           legate::int8(),
                                           legate::int16(),
                                           legate::int32(),
                                           legate::int64(),
                                           legate::uint8(),
                                           legate::uint16(),
                                           legate::uint32(),
                                           legate::uint64(),
                                           legate::float16(),
                                           legate::float32(),
                                           legate::float64(),
                                           legate::complex64(),
                                           legate::complex128()));

INSTANTIATE_TEST_SUITE_P(
  TypeUnit,
  ReductionOperatorTest,
  ::testing::Combine(
    ::testing::Values(
      legate::string_type(),
      legate::fixed_array_type(legate::int64(), 10),
      legate::struct_type(true, legate::int64(), legate::bool_(), legate::float64()),
      legate::struct_type(false, legate::int64(), legate::bool_(), legate::float64()),
      legate::binary_type(10)),
    ::testing::Values(legate::ReductionOpKind::ADD,
                      legate::ReductionOpKind::MUL,
                      legate::ReductionOpKind::MAX,
                      legate::ReductionOpKind::MIN,
                      legate::ReductionOpKind::OR,
                      legate::ReductionOpKind::AND,
                      legate::ReductionOpKind::XOR)));

INSTANTIATE_TEST_SUITE_P(TypeUnit,
                         NegativeTypeTest,
                         ::testing::Values(legate::Type::Code::FIXED_ARRAY,
                                           legate::Type::Code::STRUCT,
                                           legate::Type::Code::STRING,
                                           legate::Type::Code::LIST,
                                           legate::Type::Code::BINARY));

INSTANTIATE_TEST_SUITE_P(TypeUnit, NegativeDimTest, ::testing::Values(-1, 0, LEGATE_MAX_DIM + 1));

// NOLINTBEGIN(readability-magic-numbers)

constexpr auto GLOBAL_OP_ID = legate::GlobalRedopID{0x1F};

void test_primitive_type(const legate::Type& type,
                         const legate::Type::Code& code,
                         const std::string& type_string,
                         std::uint32_t size,
                         std::uint32_t alignment)
{
  ASSERT_EQ(type.code(), code);
  ASSERT_EQ(type.size(), size);
  ASSERT_EQ(type.alignment(), alignment);
  ASSERT_FALSE(type.variable_size());
  ASSERT_TRUE(type.is_primitive());
  ASSERT_EQ(type.to_string(), type_string);

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

void test_string_type(const legate::Type& type)
{
  ASSERT_EQ(type.code(), legate::Type::Code::STRING);
  ASSERT_THROW(static_cast<void>(type.size()), std::invalid_argument);
  ASSERT_EQ(type.alignment(), alignof(std::max_align_t));
  ASSERT_TRUE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), "string");

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

void test_binary_type(const legate::Type& type, std::uint32_t size)
{
  ASSERT_EQ(type.code(), legate::Type::Code::BINARY);
  ASSERT_EQ(type.size(), size);
  ASSERT_EQ(type.alignment(), alignof(std::max_align_t));
  ASSERT_FALSE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), fmt::format("binary({})", size));

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

void test_fixed_array_type(const legate::Type& type,
                           const legate::Type& element_type,
                           std::uint32_t N,
                           const std::string& to_string)
{
  ASSERT_EQ(type.code(), legate::Type::Code::FIXED_ARRAY);
  ASSERT_EQ(type.size(), element_type.size() * N);
  ASSERT_EQ(type.alignment(), element_type.alignment());
  ASSERT_FALSE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), to_string);

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);

  auto fixed_array_type = type.as_fixed_array_type();

  ASSERT_EQ(fixed_array_type.num_elements(), N);
  ASSERT_EQ(fixed_array_type.element_type(), element_type);
}

void test_struct_type(const legate::Type& type,
                      bool aligned,
                      std::uint32_t size,
                      std::uint32_t alignment,
                      const std::string& to_string,
                      const std::vector<legate::Type>& field_types)
{
  ASSERT_EQ(type.code(), legate::Type::Code::STRUCT);
  ASSERT_EQ(type.size(), size);
  ASSERT_EQ(type.alignment(), alignment);
  ASSERT_FALSE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);

  auto struct_type = type.as_struct_type();

  ASSERT_EQ(type.to_string(), to_string);
  ASSERT_EQ(struct_type.aligned(), aligned);
  ASSERT_EQ(struct_type.num_fields(), field_types.size());
  for (std::uint32_t idx = 0; idx < field_types.size(); ++idx) {
    ASSERT_EQ(struct_type.field_type(idx), field_types.at(idx));
  }
}

void test_list_type(const legate::Type& element_type, const std::string& to_string)
{
  auto type = legate::list_type(element_type);

  ASSERT_EQ(type.code(), legate::Type::Code::LIST);
  ASSERT_THROW((void)type.size(), std::invalid_argument);
  ASSERT_EQ(type.alignment(), 0);
  ASSERT_TRUE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);

  auto list_type = type.as_list_type();

  ASSERT_EQ(list_type.to_string(), to_string);
  ASSERT_EQ(list_type.element_type(), element_type);
}

}  // namespace

TEST_P(PrimitiveTypeFeatureTest, Basic)
{
  const auto [type, code, to_string, size, alignment] = GetParam();

  test_primitive_type(type, code, to_string, size, alignment);
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

TEST_F(TypeUnit, StringType) { test_string_type(legate::string_type()); }

TEST_P(BinaryTypeTest, Basic) { test_binary_type(legate::binary_type(GetParam()), GetParam()); }

TEST_F(TypeUnit, BinaryTypeBad)
{
  ASSERT_THROW(static_cast<void>(legate::binary_type(0xFFFFF + 1)), std::out_of_range);
}

TEST_P(FixedArrayTypeTest, Basic)
{
  const auto [element_type, size, to_string] = GetParam();
  auto fixed_array_type                      = legate::fixed_array_type(element_type, size);

  test_fixed_array_type(fixed_array_type, element_type, size, to_string);
}

TEST_F(TypeUnit, FixedArrayTypeZeroSize)
{
  // N = 0
  ASSERT_NO_THROW(static_cast<void>(legate::fixed_array_type(legate::int64(), 0)));
}

TEST_F(TypeUnit, FixedArrayTypeBadType)
{
  // element type has variable size
  ASSERT_THROW(static_cast<void>(legate::fixed_array_type(legate::string_type(), 10)),
               std::invalid_argument);
}

TEST_F(TypeUnit, FixedArrayTypeBadCast)
{
  // invalid casts
  ASSERT_THROW(static_cast<void>(legate::uint32().as_fixed_array_type()), std::invalid_argument);
}

TEST_P(StructTypeTest, Basic)
{
  const auto [type, field_types, align, total_size, alignment, to_string] = GetParam();

  test_struct_type(type, align, total_size, alignment, to_string, field_types);
}

TEST_F(TypeUnit, StructTypeBadType)
{
  // field type has variable size
  ASSERT_THROW(static_cast<void>(legate::struct_type(true, legate::string_type(), legate::int16())),
               std::runtime_error);
  ASSERT_THROW(static_cast<void>(legate::struct_type(false, legate::string_type())),
               std::runtime_error);
  ASSERT_THROW(static_cast<void>(legate::struct_type(true)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(legate::struct_type(false)), std::invalid_argument);
}

TEST_F(TypeUnit, StructTypeBadCast)
{
  // invalid casts
  ASSERT_THROW(static_cast<void>(legate::uint32().as_struct_type()), std::invalid_argument);
}

TEST_P(PointTypeTest, Basic)
{
  const auto dim = GetParam();
  auto type      = legate::point_type(dim);

  test_fixed_array_type(type, legate::int64(), dim, fmt::format("int64[{}]", dim));
  ASSERT_TRUE(legate::is_point_type(type, dim));
}

TEST_F(TypeUnit, PointTypeSpecific)
{
  // Note: There are several cases in the runtime where 64-bit integers need to be interpreted as 1D
  // points, so we need a more lenient type checking in those cases.
  ASSERT_TRUE(legate::is_point_type(legate::int64(), 1));
}

TEST_F(TypeUnit, PointTypeDimMismatch)
{
  ASSERT_FALSE(legate::is_point_type(legate::point_type(1), 2));
  ASSERT_FALSE(legate::is_point_type(legate::point_type(1), 0));
  ASSERT_FALSE(legate::is_point_type(legate::point_type(1), -1));
}

TEST_P(NegativeDimTest, PointType)
{
  ASSERT_THROW(static_cast<void>(legate::point_type(GetParam())), std::out_of_range);
}

TEST_F(TypeUnit, PointTypeBadType)
{
  ASSERT_FALSE(legate::is_point_type(legate::rect_type(1), 1));
  ASSERT_FALSE(legate::is_point_type(legate::string_type(), LEGATE_MAX_DIM));
}

TEST_P(RectTypeTest, Basic)
{
  const auto dim                              = GetParam();
  const auto type                             = legate::rect_type(dim);
  const std::vector<legate::Type> field_types = {legate::point_type(dim), legate::point_type(dim)};
  const auto full_size                        = (field_types.size() * sizeof(uint64_t)) * dim;
  const auto to_string =
    fmt::format("{{int64[{}]:0,int64[{}]:{}}}", dim, dim, dim * sizeof(std::int64_t));

  test_struct_type(type, true, full_size, sizeof(uint64_t), to_string, field_types);
  ASSERT_TRUE(legate::is_rect_type(type, dim));
}

TEST_F(TypeUnit, RectTypeDimMismatch)
{
  ASSERT_FALSE(legate::is_rect_type(legate::rect_type(1), 2));
  ASSERT_FALSE(legate::is_rect_type(legate::rect_type(1), 0));
  ASSERT_FALSE(legate::is_rect_type(legate::rect_type(1), -1));
}

TEST_P(NegativeDimTest, RectType)
{
  ASSERT_THROW(static_cast<void>(legate::rect_type(GetParam())), std::out_of_range);
}

TEST_F(TypeUnit, RectTypeBadType)
{
  ASSERT_FALSE(legate::is_rect_type(legate::point_type(1), 1));
  ASSERT_FALSE(legate::is_rect_type(legate::fixed_array_type(legate::int64(), 1), 1));
  ASSERT_FALSE(legate::is_rect_type(legate::int64(), 1));
}

TEST_P(ListTypeTest, Basic)
{
  const auto [type, to_string] = GetParam();

  test_list_type(type, to_string);
}

TEST_F(TypeUnit, ListTypeBadType)
{
  // variable size types
  ASSERT_THROW(static_cast<void>(legate::list_type(legate::string_type())), std::runtime_error);
  ASSERT_THROW(static_cast<void>(legate::list_type(legate::list_type(legate::uint32()))),
               std::runtime_error);
}

TEST_F(TypeUnit, ListTypeBadCast)
{
  // invald casts
  ASSERT_THROW(static_cast<void>(legate::string_type().as_struct_type()), std::invalid_argument);
}

TEST_P(UidFixedArrayTypeTest, PrimitiveType)
{
  const auto [element_type, dim] = GetParam();
  auto fixed_array_type          = legate::fixed_array_type(element_type, dim);

  ASSERT_EQ(fixed_array_type.uid() & 0x00FF, static_cast<std::int32_t>(element_type.code()));
  ASSERT_EQ(fixed_array_type.uid() >> 8, dim);
  ASSERT_EQ(fixed_array_type.as_fixed_array_type().num_elements(), dim);

  ASSERT_NE(fixed_array_type.uid(), legate::string_type().uid());
  ASSERT_NE(fixed_array_type.uid(), static_cast<std::int32_t>(element_type.code()));

  auto same_array_type = legate::fixed_array_type(element_type, dim);

  ASSERT_EQ(fixed_array_type.uid(), same_array_type.uid());

  auto diff_array_type = legate::fixed_array_type(element_type, (dim % LEGATE_MAX_DIM) + 1);

  ASSERT_NE(fixed_array_type.uid(), diff_array_type.uid());
}

TEST_P(UidFixedArrayTypeTest, FixedArrayType)
{
  const auto [element_type, dim] = GetParam();
  // array of array types
  auto array_of_array_type1 =
    legate::fixed_array_type(legate::fixed_array_type(element_type, dim), dim);
  auto array_of_array_type2 =
    legate::fixed_array_type(legate::fixed_array_type(element_type, dim), dim);

  ASSERT_NE(array_of_array_type1.uid(), array_of_array_type2.uid());
}

TEST_P(UidFixedArrayTypeTest, PointType)
{
  const auto [element_type, dim] = GetParam();
  // array of point types
  auto array_of_point_type1 = legate::fixed_array_type(legate::point_type(dim), dim);
  auto array_of_point_type2 = legate::fixed_array_type(legate::point_type(dim), dim);

  ASSERT_NE(array_of_point_type1.uid(), array_of_point_type2.uid());
}

TEST_P(UidFixedArrayTypeTest, RectType)
{
  const auto [element_type, dim] = GetParam();
  // array of rect types
  auto array_of_rect_type1 = legate::fixed_array_type(legate::rect_type(dim), dim);
  auto array_of_rect_type2 = legate::fixed_array_type(legate::rect_type(dim), dim);

  ASSERT_NE(array_of_rect_type1.uid(), array_of_rect_type2.uid());
}

TEST_P(UidFixedArrayTypeTest, StructType)
{
  const auto [element_type, dim] = GetParam();
  // array of struct types
  auto array_of_struct_type1 =
    legate::fixed_array_type(legate::struct_type(true, element_type), dim);
  auto array_of_struct_type2 =
    legate::fixed_array_type(legate::struct_type(true, element_type), dim);

  ASSERT_NE(array_of_struct_type1.uid(), array_of_struct_type2.uid());
}

TEST_P(UidTest, LargeSizeArray)
{
  // N > 0xFFU
  auto big_array_type = legate::fixed_array_type(GetParam(), 300);

  ASSERT_TRUE(big_array_type.uid() >= 0x10000);
}

TEST_F(TypeUnit, StringTypeUid)
{
  ASSERT_EQ(legate::string_type().uid(), static_cast<std::int32_t>(legate::Type::Code::STRING));
}

TEST_P(UidTest, StructType)
{
  const auto element_type = GetParam();
  auto struct_type1       = legate::struct_type(true, element_type);
  auto struct_type2       = legate::struct_type(true, element_type);

  ASSERT_NE(struct_type1.uid(), struct_type2.uid());
  ASSERT_TRUE(struct_type1.uid() >= 0x10000);
  ASSERT_TRUE(struct_type2.uid() >= 0x10000);

  ASSERT_NE(struct_type1.uid(), legate::string_type().uid());
  ASSERT_NE(struct_type2.uid(), legate::string_type().uid());
  ASSERT_NE(struct_type1.uid(), static_cast<std::int32_t>(element_type.code()));
  ASSERT_NE(struct_type2.uid(), static_cast<std::int32_t>(element_type.code()));
}

TEST_P(UidTest, ListType)
{
  const auto element_type = GetParam();
  auto list_type1         = legate::list_type(element_type);
  auto list_type2         = legate::list_type(element_type);

  ASSERT_NE(list_type1.uid(), list_type2.uid());
  ASSERT_TRUE(list_type1.uid() >= 0x10000);
  ASSERT_TRUE(list_type2.uid() >= 0x10000);

  ASSERT_NE(list_type1.uid(), legate::string_type().uid());
  ASSERT_NE(list_type2.uid(), legate::string_type().uid());
  ASSERT_NE(list_type1.uid(), static_cast<std::int32_t>(element_type.code()));
  ASSERT_NE(list_type2.uid(), static_cast<std::int32_t>(element_type.code()));
}

TEST_F(TypeUnit, BinaryTypeUid)
{
  auto binary_type      = legate::binary_type(678);
  auto same_binary_type = legate::binary_type(678);
  auto diff_binary_type = legate::binary_type(67);

  ASSERT_EQ(binary_type.uid(), same_binary_type.uid());
  ASSERT_NE(binary_type.uid(), diff_binary_type.uid());
}

TEST_P(ReductionOperatorTest, Record)
{
  const auto [type, op_kind] = GetParam();

  type.record_reduction_operator(static_cast<std::int32_t>(op_kind), GLOBAL_OP_ID);
  ASSERT_EQ(type.find_reduction_operator(static_cast<std::int32_t>(op_kind)), GLOBAL_OP_ID);
  ASSERT_EQ(type.find_reduction_operator(op_kind), GLOBAL_OP_ID);

  // repeat records for same type
  ASSERT_THROW(type.record_reduction_operator(static_cast<std::int32_t>(op_kind), GLOBAL_OP_ID),
               std::invalid_argument);
}

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

// NOLINTEND(readability-magic-numbers)

}  // namespace type_test
