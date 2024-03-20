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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace type_test {

using TypeUnit = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

constexpr std::int32_t GLOBAL_OP_ID = 0x1F;

const std::array<legate::Type, 13>& PRIMITIVE_TYPE()
{
  static const std::array<legate::Type, 13> arr = {legate::bool_(),
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
                                                   legate::complex128()};

  return arr;
}

template <typename T>
struct alignment_of : std::integral_constant<std::size_t, alignof(T)> {};

template <>
struct alignment_of<void> : std::integral_constant<std::size_t, 0> {};

template <typename T>
void test_primitive_type(const legate::Type& type,
                         const std::string& type_string,
                         std::uint32_t size)
{
  EXPECT_EQ(type.code(), legate::type_code_of<T>);
  EXPECT_EQ(type.size(), size);
  // need extra layer of template indirection since alignof(void) (for null_type) is illegal
  EXPECT_EQ(type.alignment(), alignment_of<T>::value);
  EXPECT_FALSE(type.variable_size());
  EXPECT_TRUE(type.is_primitive());
  EXPECT_EQ(type.to_string(), type_string);

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(other, type);
}

void test_string_type(const legate::Type& type)
{
  EXPECT_EQ(type.code(), legate::Type::Code::STRING);
  EXPECT_THROW(static_cast<void>(type.size()), std::invalid_argument);
  EXPECT_EQ(type.alignment(), alignof(std::max_align_t));
  EXPECT_TRUE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  EXPECT_EQ(type.to_string(), "string");

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(other, type);
}

void test_binary_type(const legate::Type& type, std::uint32_t size)
{
  EXPECT_EQ(type.code(), legate::Type::Code::BINARY);
  EXPECT_EQ(type.size(), size);
  EXPECT_EQ(type.alignment(), alignof(std::max_align_t));
  EXPECT_FALSE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  EXPECT_EQ(type.to_string(), "binary(" + std::to_string(size) + ")");

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(other, type);
}

void test_fixed_array_type(const legate::Type& type,
                           const legate::Type& element_type,
                           std::uint32_t N,
                           const std::string& to_string)
{
  EXPECT_EQ(type.code(), legate::Type::Code::FIXED_ARRAY);
  EXPECT_EQ(type.size(), element_type.size() * N);
  EXPECT_EQ(type.alignment(), element_type.alignment());
  EXPECT_FALSE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  EXPECT_EQ(type.to_string(), to_string);

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(other, type);

  auto fixed_array_type = type.as_fixed_array_type();
  EXPECT_EQ(fixed_array_type.num_elements(), N);
  EXPECT_EQ(fixed_array_type.element_type(), element_type);
}

void test_struct_type(const legate::Type& type,
                      bool aligned,
                      std::uint32_t size,
                      std::uint32_t alignment,
                      const std::string& to_string,
                      const std::vector<legate::Type>& field_types,
                      const std::vector<std::uint32_t>& /*offsets*/)
{
  EXPECT_EQ(type.code(), legate::Type::Code::STRUCT);
  EXPECT_EQ(type.size(), size);
  EXPECT_EQ(type.alignment(), alignment);
  EXPECT_FALSE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(other, type);

  auto struct_type = type.as_struct_type();
  EXPECT_EQ(type.to_string(), to_string);
  EXPECT_EQ(struct_type.aligned(), aligned);
  EXPECT_EQ(struct_type.num_fields(), field_types.size());
  for (std::uint32_t idx = 0; idx < field_types.size(); ++idx) {
    EXPECT_EQ(struct_type.field_type(idx), field_types.at(idx));
  }
}

void test_list_type(const legate::Type& element_type, const std::string& to_string)
{
  auto type = legate::list_type(element_type);
  EXPECT_EQ(type.code(), legate::Type::Code::LIST);
  EXPECT_THROW((void)type.size(), std::invalid_argument);
  EXPECT_EQ(type.alignment(), 0);
  EXPECT_TRUE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());

  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(other, type);

  auto list_type = type.as_list_type();
  EXPECT_EQ(list_type.to_string(), to_string);
  EXPECT_EQ(list_type.element_type(), element_type);
}

void test_reduction_op(const legate::Type& type)
{
  type.record_reduction_operator(static_cast<std::int32_t>(legate::ReductionOpKind::ADD),
                                 GLOBAL_OP_ID);
  EXPECT_EQ(type.find_reduction_operator(static_cast<std::int32_t>(legate::ReductionOpKind::ADD)),
            GLOBAL_OP_ID);
  EXPECT_EQ(type.find_reduction_operator(legate::ReductionOpKind::ADD), GLOBAL_OP_ID);

  // repeat records for same type
  EXPECT_THROW(type.record_reduction_operator(
                 static_cast<std::int32_t>(legate::ReductionOpKind::ADD), GLOBAL_OP_ID),
               std::invalid_argument);

  // reduction op doesn't exist
  EXPECT_THROW(
    (void)type.find_reduction_operator(static_cast<std::int32_t>(legate::ReductionOpKind::SUB)),
    std::invalid_argument);
  EXPECT_THROW((void)type.find_reduction_operator(legate::ReductionOpKind::SUB),
               std::invalid_argument);
}

TEST_F(TypeUnit, PrimitiveType)
{
  test_primitive_type<void>(legate::null_type(), "null_type", 0);
  test_primitive_type<bool>(legate::bool_(), "bool", sizeof(bool));
  test_primitive_type<std::int8_t>(legate::int8(), "int8", sizeof(int8_t));
  test_primitive_type<std::int16_t>(legate::int16(), "int16", sizeof(int16_t));
  test_primitive_type<std::int32_t>(legate::int32(), "int32", sizeof(int32_t));
  test_primitive_type<std::int64_t>(legate::int64(), "int64", sizeof(int64_t));
  test_primitive_type<std::uint8_t>(legate::uint8(), "uint8", sizeof(uint8_t));
  test_primitive_type<std::uint16_t>(legate::uint16(), "uint16", sizeof(uint16_t));
  test_primitive_type<std::uint32_t>(legate::uint32(), "uint32", sizeof(uint32_t));
  test_primitive_type<std::uint64_t>(legate::uint64(), "uint64", sizeof(uint64_t));
  test_primitive_type<__half>(legate::float16(), "float16", sizeof(__half));
  test_primitive_type<float>(legate::float32(), "float32", sizeof(float));
  test_primitive_type<double>(legate::float64(), "float64", sizeof(double));
  test_primitive_type<complex<float>>(legate::complex64(), "complex64", sizeof(complex<float>));
  test_primitive_type<complex<double>>(legate::complex128(), "complex128", sizeof(complex<double>));

  EXPECT_EQ(legate::null_type(), legate::primitive_type(legate::Type::Code::NIL));
  EXPECT_EQ(legate::bool_(), legate::primitive_type(legate::Type::Code::BOOL));
  EXPECT_EQ(legate::int8(), legate::primitive_type(legate::Type::Code::INT8));
  EXPECT_EQ(legate::int16(), legate::primitive_type(legate::Type::Code::INT16));
  EXPECT_EQ(legate::int32(), legate::primitive_type(legate::Type::Code::INT32));
  EXPECT_EQ(legate::int64(), legate::primitive_type(legate::Type::Code::INT64));
  EXPECT_EQ(legate::uint8(), legate::primitive_type(legate::Type::Code::UINT8));
  EXPECT_EQ(legate::uint16(), legate::primitive_type(legate::Type::Code::UINT16));
  EXPECT_EQ(legate::uint32(), legate::primitive_type(legate::Type::Code::UINT32));
  EXPECT_EQ(legate::uint64(), legate::primitive_type(legate::Type::Code::UINT64));
  EXPECT_EQ(legate::float16(), legate::primitive_type(legate::Type::Code::FLOAT16));
  EXPECT_EQ(legate::float32(), legate::primitive_type(legate::Type::Code::FLOAT32));
  EXPECT_EQ(legate::float64(), legate::primitive_type(legate::Type::Code::FLOAT64));
  EXPECT_EQ(legate::complex64(), legate::primitive_type(legate::Type::Code::COMPLEX64));
  EXPECT_EQ(legate::complex128(), legate::primitive_type(legate::Type::Code::COMPLEX128));

  EXPECT_THROW((void)legate::primitive_type(legate::Type::Code::FIXED_ARRAY),
               std::invalid_argument);
  EXPECT_THROW((void)legate::primitive_type(legate::Type::Code::STRUCT), std::invalid_argument);
  EXPECT_THROW((void)legate::primitive_type(legate::Type::Code::STRING), std::invalid_argument);
  EXPECT_THROW((void)legate::primitive_type(legate::Type::Code::LIST), std::invalid_argument);
  EXPECT_THROW((void)legate::primitive_type(legate::Type::Code::BINARY), std::invalid_argument);
}

TEST_F(TypeUnit, StringType) { test_string_type(legate::string_type()); }

TEST_F(TypeUnit, BinaryType)
{
  test_binary_type(legate::binary_type(123), 123);
  test_binary_type(legate::binary_type(45), 45);

  EXPECT_THROW((void)legate::binary_type(0), std::out_of_range);
  EXPECT_THROW((void)legate::binary_type(0xFFFFF + 1), std::out_of_range);
}

TEST_F(TypeUnit, FixedArrayType)
{
  // element type is a primitive type
  {
    constexpr std::uint32_t N = 10;
    auto element_type         = legate::uint64();
    auto fixed_array_type     = legate::fixed_array_type(element_type, N);
    test_fixed_array_type(fixed_array_type, element_type, N, "uint64[10]");
  }

  // element type is not a primitive type
  {
    constexpr std::uint32_t N = 10;
    auto element_type         = legate::fixed_array_type(legate::uint16(), N);
    auto fixed_array_type     = legate::fixed_array_type(element_type, N);
    test_fixed_array_type(fixed_array_type, element_type, N, "uint16[10][10]");
  }

  // N > 0xFFU
  {
    constexpr std::uint32_t N = 256;
    auto element_type         = legate::float64();
    auto fixed_array_type     = legate::fixed_array_type(element_type, N);
    test_fixed_array_type(fixed_array_type, element_type, N, "float64[256]");
  }

  // N = 0
  {
    EXPECT_THROW((void)legate::fixed_array_type(legate::int64(), 0), std::out_of_range);
  }

  // element type has variable size
  EXPECT_THROW((void)legate::fixed_array_type(legate::string_type(), 10), std::invalid_argument);

  // invalid casts
  EXPECT_THROW((void)legate::uint32().as_fixed_array_type(), std::invalid_argument);
}

TEST_F(TypeUnit, StructType)
{
  // aligned
  {
    const auto type =
      legate::struct_type(true, legate::int16(), legate::bool_(), legate::float64());
    const std::vector<legate::Type> field_types = {
      legate::int16(), legate::bool_(), legate::float64()};
    const std::vector<std::uint32_t> offsets = {0, 2, 8};
    constexpr auto size                      = 16;  //  size = 8 (std::int16_t bool) + 8 (double)

    test_struct_type(
      type, true, size, sizeof(double), "{int16:0,bool:2,float64:8}", field_types, offsets);
  }

  // aligned
  {
    const std::vector<legate::Type> field_types = {
      legate::bool_(), legate::float64(), legate::int16()};
    const auto type                          = legate::struct_type(field_types, true);
    const std::vector<std::uint32_t> offsets = {0, 8, 16};
    constexpr auto size = 24;  // size: 24 = 8 (bool) + 8 (double) + 8 (int16_t)

    test_struct_type(
      type, true, size, sizeof(double), "{bool:0,float64:8,int16:16}", field_types, offsets);
  }

  // not aligned
  {
    const auto type =
      legate::struct_type(false, legate::int16(), legate::bool_(), legate::float64());
    const std::vector<legate::Type> field_types = {
      legate::int16(), legate::bool_(), legate::float64()};
    const std::vector<std::uint32_t> offsets = {0, 2, 3};
    constexpr auto size                      = sizeof(int16_t) + sizeof(bool) + sizeof(double);

    test_struct_type(type, false, size, 1, "{int16:0,bool:2,float64:3}", field_types, offsets);
  }

  // not aligned
  {
    const std::vector<legate::Type> field_types = {
      legate::bool_(), legate::float64(), legate::int16()};
    const auto type                          = legate::struct_type(field_types);
    const std::vector<std::uint32_t> offsets = {0, 1, 9};
    constexpr auto size                      = sizeof(int16_t) + sizeof(bool) + sizeof(double);

    test_struct_type(type, false, size, 1, "{bool:0,float64:1,int16:9}", field_types, offsets);
  }

  // field type has variable size
  EXPECT_THROW((void)legate::struct_type(true, legate::string_type(), legate::int16()),
               std::runtime_error);
  EXPECT_THROW((void)legate::struct_type(false, legate::string_type()), std::runtime_error);
  EXPECT_THROW((void)legate::struct_type(true), std::invalid_argument);
  EXPECT_THROW((void)legate::struct_type(false), std::invalid_argument);

  // invalid casts
  EXPECT_THROW((void)legate::uint32().as_struct_type(), std::invalid_argument);
}

TEST_F(TypeUnit, PointType)
{
  for (std::uint32_t idx = 1; idx <= LEGATE_MAX_DIM; ++idx) {
    auto type = legate::point_type(idx);

    test_fixed_array_type(type, legate::int64(), idx, "int64[" + std::to_string(idx) + "]");
    EXPECT_TRUE(legate::is_point_type(type, idx));
  }

  // point type checks
  EXPECT_FALSE(legate::is_point_type(legate::point_type(1), 2));
  EXPECT_FALSE(legate::is_point_type(legate::point_type(1), 0));
  EXPECT_FALSE(legate::is_point_type(legate::point_type(1), -1));

  EXPECT_FALSE(legate::is_point_type(legate::rect_type(1), 1));
  EXPECT_FALSE(legate::is_point_type(legate::string_type(), 4));
  // Note: There are several cases in the runtime where 64-bit integers need to be interpreted as 1D
  // points, so we need a more lenient type checking in those cases.
  EXPECT_TRUE(legate::is_point_type(legate::int64(), 1));

  // invalid dim
  EXPECT_THROW((void)legate::point_type(-1), std::out_of_range);
  EXPECT_THROW((void)legate::point_type(0), std::out_of_range);
  EXPECT_THROW((void)legate::point_type(LEGATE_MAX_DIM + 1), std::out_of_range);
}

TEST_F(TypeUnit, RectType)
{
  for (std::uint32_t idx = 1; idx <= LEGATE_MAX_DIM; ++idx) {
    const auto type                             = legate::rect_type(idx);
    const std::vector<legate::Type> field_types = {legate::point_type(idx),
                                                   legate::point_type(idx)};
    const std::vector<std::uint32_t> offsets    = {0,
                                                   static_cast<std::uint32_t>(sizeof(uint64_t)) * idx};
    const auto full_size                        = (field_types.size() * sizeof(uint64_t)) * idx;
    const auto to_string = "{int64[" + std::to_string(idx) + "]:0,int64[" + std::to_string(idx) +
                           "]:" + std::to_string(idx * sizeof(int64_t)) + "}";

    test_struct_type(type, true, full_size, sizeof(uint64_t), to_string, field_types, offsets);
    EXPECT_TRUE(legate::is_rect_type(type, idx));
  }

  // rect type checks
  EXPECT_FALSE(legate::is_rect_type(legate::rect_type(1), 2));
  EXPECT_FALSE(legate::is_rect_type(legate::rect_type(1), 0));
  EXPECT_FALSE(legate::is_rect_type(legate::rect_type(1), -1));

  EXPECT_FALSE(legate::is_rect_type(legate::point_type(1), 1));
  EXPECT_FALSE(legate::is_rect_type(legate::fixed_array_type(legate::int64(), 1), 1));
  EXPECT_FALSE(legate::is_rect_type(legate::int64(), 1));

  // invalid dim
  EXPECT_THROW((void)legate::rect_type(-1), std::out_of_range);
  EXPECT_THROW((void)legate::rect_type(0), std::out_of_range);
  EXPECT_THROW((void)legate::rect_type(LEGATE_MAX_DIM + 1), std::out_of_range);
}

TEST_F(TypeUnit, ListType)
{
  test_list_type(legate::bool_(), "list(bool)");
  test_list_type(legate::int8(), "list(int8)");
  test_list_type(legate::int16(), "list(int16)");
  test_list_type(legate::int32(), "list(int32)");
  test_list_type(legate::int64(), "list(int64)");
  test_list_type(legate::uint8(), "list(uint8)");
  test_list_type(legate::uint16(), "list(uint16)");
  test_list_type(legate::uint32(), "list(uint32)");
  test_list_type(legate::uint64(), "list(uint64)");
  test_list_type(legate::float16(), "list(float16)");
  test_list_type(legate::float32(), "list(float32)");
  test_list_type(legate::float64(), "list(float64)");
  test_list_type(legate::complex64(), "list(complex64)");
  test_list_type(legate::complex128(), "list(complex128)");
  test_list_type(legate::struct_type(true, legate::bool_(), legate::int32()),
                 "list({bool:0,int32:4})");
  test_list_type(legate::struct_type(false, legate::bool_(), legate::int32()),
                 "list({bool:0,int32:1})");
  test_list_type(legate::point_type(1), "list(int64[1])");
  test_list_type(legate::rect_type(2), "list({int64[2]:0,int64[2]:16})");

  // variable size types
  EXPECT_THROW((void)legate::list_type(legate::string_type()), std::runtime_error);
  EXPECT_THROW((void)legate::list_type(legate::list_type(legate::uint32())), std::runtime_error);

  // invald casts
  EXPECT_THROW((void)legate::string_type().as_struct_type(), std::invalid_argument);
}

TEST_F(TypeUnit, Uid)
{
  // fixed array type
  for (std::uint32_t idx = 0; idx < PRIMITIVE_TYPE().size(); ++idx) {
    auto element_type     = PRIMITIVE_TYPE().at(idx);
    const auto N          = idx + 1;
    auto fixed_array_type = legate::fixed_array_type(element_type, N);

    EXPECT_EQ(fixed_array_type.uid() & 0x00FF, static_cast<std::int32_t>(element_type.code()));
    EXPECT_EQ(fixed_array_type.uid() >> 8, N);
    EXPECT_EQ(fixed_array_type.as_fixed_array_type().num_elements(), N);

    EXPECT_NE(fixed_array_type.uid(), legate::string_type().uid());
    EXPECT_NE(fixed_array_type.uid(), static_cast<std::int32_t>(element_type.code()));

    auto same_array_type = legate::fixed_array_type(element_type, N);
    EXPECT_EQ(fixed_array_type.uid(), same_array_type.uid());

    auto diff_array_type = legate::fixed_array_type(element_type, N + 1);
    EXPECT_NE(fixed_array_type.uid(), diff_array_type.uid());

    // N > 0xFFU
    auto big_array_type = legate::fixed_array_type(element_type, 300);
    EXPECT_TRUE(big_array_type.uid() >= 0x10000);

    // array of array types
    auto array_of_array_type1 =
      legate::fixed_array_type(legate::fixed_array_type(element_type, N), N);
    auto array_of_array_type2 =
      legate::fixed_array_type(legate::fixed_array_type(element_type, N), N);
    EXPECT_NE(array_of_array_type1.uid(), array_of_array_type2.uid());

    // array of point types
    const auto dim            = N % LEGATE_MAX_DIM + 1;
    auto array_of_point_type1 = legate::fixed_array_type(legate::point_type(dim), N);
    auto array_of_point_type2 = legate::fixed_array_type(legate::point_type(dim), N);
    EXPECT_NE(array_of_point_type1.uid(), array_of_point_type2.uid());

    // array of rect types
    auto array_of_rect_type1 = legate::fixed_array_type(legate::rect_type(dim), N);
    auto array_of_rect_type2 = legate::fixed_array_type(legate::rect_type(dim), N);
    EXPECT_NE(array_of_rect_type1.uid(), array_of_rect_type2.uid());

    // array of struct types
    auto array_of_struct_type1 =
      legate::fixed_array_type(legate::struct_type(true, element_type), N);
    auto array_of_struct_type2 =
      legate::fixed_array_type(legate::struct_type(true, element_type), N);
    EXPECT_NE(array_of_struct_type1.uid(), array_of_struct_type2.uid());
  }

  // string type
  EXPECT_EQ(legate::string_type().uid(), static_cast<std::int32_t>(legate::Type::Code::STRING));

  // struct type
  for (auto&& element_type : PRIMITIVE_TYPE()) {
    auto struct_type1 = legate::struct_type(true, element_type);
    auto struct_type2 = legate::struct_type(true, element_type);

    EXPECT_NE(struct_type1.uid(), struct_type2.uid());
    EXPECT_TRUE(struct_type1.uid() >= 0x10000);
    EXPECT_TRUE(struct_type2.uid() >= 0x10000);

    EXPECT_NE(struct_type1.uid(), legate::string_type().uid());
    EXPECT_NE(struct_type2.uid(), legate::string_type().uid());
    EXPECT_NE(struct_type1.uid(), static_cast<std::int32_t>(element_type.code()));
    EXPECT_NE(struct_type2.uid(), static_cast<std::int32_t>(element_type.code()));
  }

  // list type
  for (auto&& element_type : PRIMITIVE_TYPE()) {
    auto list_type1 = legate::list_type(element_type);
    auto list_type2 = legate::list_type(element_type);

    EXPECT_NE(list_type1.uid(), list_type2.uid());
    EXPECT_TRUE(list_type1.uid() >= 0x10000);
    EXPECT_TRUE(list_type2.uid() >= 0x10000);

    EXPECT_NE(list_type1.uid(), legate::string_type().uid());
    EXPECT_NE(list_type2.uid(), legate::string_type().uid());
    EXPECT_NE(list_type1.uid(), static_cast<std::int32_t>(element_type.code()));
    EXPECT_NE(list_type2.uid(), static_cast<std::int32_t>(element_type.code()));
  }

  // binary type
  auto binary_type      = legate::binary_type(678);
  auto same_binary_type = legate::binary_type(678);
  auto diff_binary_type = legate::binary_type(67);

  EXPECT_EQ(binary_type.uid(), same_binary_type.uid());
  EXPECT_NE(binary_type.uid(), diff_binary_type.uid());
}

TEST_F(TypeUnit, ReductionOperator)
{
  test_reduction_op(legate::string_type());
  test_reduction_op(legate::fixed_array_type(legate::int64(), 10));
  test_reduction_op(legate::struct_type(true, legate::int64(), legate::bool_(), legate::float64()));
  test_reduction_op(
    legate::struct_type(false, legate::int64(), legate::bool_(), legate::float64()));
  test_reduction_op(legate::binary_type(10));
}

TEST_F(TypeUnit, TypeCodeOf)
{
  EXPECT_EQ(legate::type_code_of<void>, legate::Type::Code::NIL);
  EXPECT_EQ(legate::type_code_of<bool>, legate::Type::Code::BOOL);
  EXPECT_EQ(legate::type_code_of<std::int8_t>, legate::Type::Code::INT8);
  EXPECT_EQ(legate::type_code_of<std::int16_t>, legate::Type::Code::INT16);
  EXPECT_EQ(legate::type_code_of<std::int32_t>, legate::Type::Code::INT32);
  EXPECT_EQ(legate::type_code_of<std::int64_t>, legate::Type::Code::INT64);
  EXPECT_EQ(legate::type_code_of<std::uint8_t>, legate::Type::Code::UINT8);
  EXPECT_EQ(legate::type_code_of<std::uint16_t>, legate::Type::Code::UINT16);
  EXPECT_EQ(legate::type_code_of<std::uint32_t>, legate::Type::Code::UINT32);
  EXPECT_EQ(legate::type_code_of<std::uint64_t>, legate::Type::Code::UINT64);
  EXPECT_EQ(legate::type_code_of<__half>, legate::Type::Code::FLOAT16);
  EXPECT_EQ(legate::type_code_of<float>, legate::Type::Code::FLOAT32);
  EXPECT_EQ(legate::type_code_of<double>, legate::Type::Code::FLOAT64);
  EXPECT_EQ(legate::type_code_of<complex<float>>, legate::Type::Code::COMPLEX64);
  EXPECT_EQ(legate::type_code_of<complex<double>>, legate::Type::Code::COMPLEX128);
  EXPECT_EQ(legate::type_code_of<std::string>, legate::Type::Code::STRING);
}

TEST_F(TypeUnit, TypeOf)
{
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::BOOL>, bool>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::INT8>, std::int8_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::INT16>, std::int16_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::INT32>, std::int32_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::INT64>, std::int64_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::UINT8>, std::uint8_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::UINT16>, std::uint16_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::UINT32>, std::uint32_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::UINT64>, std::uint64_t>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::FLOAT16>, __half>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::FLOAT32>, float>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::FLOAT64>, double>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::COMPLEX64>, complex<float>>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::COMPLEX128>, complex<double>>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::STRING>, std::string>));

  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::NIL>, void>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::FIXED_ARRAY>, void>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::STRUCT>, void>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::LIST>, void>));
  EXPECT_TRUE((std::is_same_v<legate::type_of<legate::Type::Code::BINARY>, void>));
}

TEST_F(TypeUnit, TypeUtils)
{
  // is_integral
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::BOOL>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::INT8>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::INT16>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::INT32>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::INT64>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::UINT8>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::UINT16>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::UINT32>::value);
  EXPECT_TRUE(legate::is_integral<legate::Type::Code::UINT64>::value);

  EXPECT_FALSE(legate::is_integral<legate::Type::Code::FLOAT16>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::FLOAT32>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::FLOAT64>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::COMPLEX64>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::COMPLEX128>::value);

  EXPECT_FALSE(legate::is_integral<legate::Type::Code::NIL>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::LIST>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::BINARY>::value);

  // is_signed
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::INT8>::value);
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::INT16>::value);
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::INT32>::value);
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::INT64>::value);
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::FLOAT32>::value);
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::FLOAT64>::value);
  EXPECT_TRUE(legate::is_signed<legate::Type::Code::FLOAT16>::value);

  EXPECT_FALSE(legate::is_signed<legate::Type::Code::BOOL>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::UINT8>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::UINT16>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::UINT32>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::UINT64>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::COMPLEX64>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::COMPLEX128>::value);

  EXPECT_FALSE(legate::is_signed<legate::Type::Code::NIL>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::LIST>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::BINARY>::value);

  // is_unsigned
  EXPECT_TRUE(legate::is_unsigned<legate::Type::Code::BOOL>::value);
  EXPECT_TRUE(legate::is_unsigned<legate::Type::Code::UINT8>::value);
  EXPECT_TRUE(legate::is_unsigned<legate::Type::Code::UINT16>::value);
  EXPECT_TRUE(legate::is_unsigned<legate::Type::Code::UINT32>::value);
  EXPECT_TRUE(legate::is_unsigned<legate::Type::Code::UINT64>::value);

  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::INT8>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::INT16>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::INT32>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::INT64>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::FLOAT16>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::FLOAT32>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::FLOAT64>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::COMPLEX64>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::COMPLEX128>::value);

  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::NIL>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::LIST>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::BINARY>::value);

  // is_floating_point
  EXPECT_TRUE(legate::is_floating_point<legate::Type::Code::FLOAT16>::value);
  EXPECT_TRUE(legate::is_floating_point<legate::Type::Code::FLOAT32>::value);
  EXPECT_TRUE(legate::is_floating_point<legate::Type::Code::FLOAT64>::value);

  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::BOOL>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::UINT8>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::UINT16>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::UINT32>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::UINT64>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::INT8>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::INT16>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::INT32>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::INT64>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::COMPLEX64>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::COMPLEX128>::value);

  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::NIL>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::LIST>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::BINARY>::value);

  // is_complex
  EXPECT_TRUE(legate::is_complex<legate::Type::Code::COMPLEX64>::value);
  EXPECT_TRUE(legate::is_complex<legate::Type::Code::COMPLEX128>::value);

  EXPECT_FALSE(legate::is_complex<legate::Type::Code::BOOL>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::UINT8>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::UINT16>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::UINT32>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::UINT64>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::INT8>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::INT16>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::INT32>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::INT64>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::FLOAT16>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::FLOAT32>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::FLOAT64>::value);

  EXPECT_FALSE(legate::is_complex<legate::Type::Code::NIL>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::LIST>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::BINARY>::value);

  // is_complex_type
  EXPECT_TRUE(legate::is_complex_type<complex<float>>::value);
  EXPECT_TRUE(legate::is_complex_type<complex<double>>::value);

  EXPECT_FALSE(legate::is_complex_type<bool>::value);
  EXPECT_FALSE(legate::is_complex_type<std::int8_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::int16_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::int32_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::int64_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::uint8_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::uint16_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::uint32_t>::value);
  EXPECT_FALSE(legate::is_complex_type<std::uint64_t>::value);
  EXPECT_FALSE(legate::is_complex_type<__half>::value);
  EXPECT_FALSE(legate::is_complex_type<float>::value);
  EXPECT_FALSE(legate::is_complex_type<double>::value);

  EXPECT_FALSE(legate::is_complex_type<void>::value);
  EXPECT_FALSE(legate::is_complex_type<std::string>::value);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace type_test
