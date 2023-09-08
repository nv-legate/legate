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

#include <gtest/gtest.h>
#include "legate.h"

namespace type_test {
constexpr int32_t GLOBAL_OP_ID = 0x1F;

const std::vector<legate::Type> PRIMITIVE_TYPE = {legate::bool_(),
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

template <typename T>
void test_primitive_type(const legate::Type& type, std::string type_string)
{
  EXPECT_EQ(type.code(), legate::legate_type_code_of<T>);
  EXPECT_EQ(type.size(), sizeof(T));
  EXPECT_EQ(type.alignment(), sizeof(T));
  EXPECT_FALSE(type.variable_size());
  EXPECT_TRUE(type.is_primitive());
  EXPECT_EQ(type.to_string(), type_string);
  legate::Type other(type);
  EXPECT_EQ(other, type);
}

void test_string_type(const legate::Type& type)
{
  EXPECT_EQ(type.code(), legate::Type::Code::STRING);
  EXPECT_THROW(type.size(), std::invalid_argument);
  EXPECT_EQ(type.alignment(), 0);
  EXPECT_TRUE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  EXPECT_EQ(type.to_string(), "string");
  legate::Type other(type);
  EXPECT_EQ(other, type);
}

void test_fixed_array_type(const legate::Type& type,
                           const legate::Type& element_type,
                           uint32_t N,
                           std::string to_string)
{
  EXPECT_EQ(type.code(), legate::Type::Code::FIXED_ARRAY);
  EXPECT_EQ(type.size(), element_type.size() * N);
  EXPECT_EQ(type.alignment(), element_type.alignment());
  EXPECT_FALSE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  EXPECT_EQ(type.to_string(), to_string);
  legate::Type other(type);
  EXPECT_EQ(other, type);

  auto fixed_array_type = type.as_fixed_array_type();
  EXPECT_EQ(fixed_array_type.num_elements(), N);
  EXPECT_EQ(fixed_array_type.element_type(), element_type);
}

void test_struct_type(const legate::Type& type,
                      bool aligned,
                      uint32_t size,
                      uint32_t alignment,
                      std::string to_string,
                      std::vector<legate::Type> field_types,
                      std::vector<uint32_t> offsets)
{
  EXPECT_EQ(type.code(), legate::Type::Code::STRUCT);
  EXPECT_EQ(type.size(), size);
  EXPECT_EQ(type.alignment(), alignment);
  EXPECT_FALSE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  legate::Type other(type);
  EXPECT_EQ(other, type);

  auto struct_type = type.as_struct_type();
  EXPECT_EQ(type.to_string(), to_string);
  EXPECT_EQ(struct_type.aligned(), aligned);
  EXPECT_EQ(struct_type.num_fields(), field_types.size());
  for (uint32_t idx = 0; idx < field_types.size(); ++idx) {
    EXPECT_EQ(struct_type.field_type(idx), field_types.at(idx));
  }
}

void test_list_type(const legate::Type& element_type, std::string to_string)
{
  auto type = legate::list_type(element_type);
  EXPECT_EQ(type.code(), legate::Type::Code::LIST);
  EXPECT_THROW(type.size(), std::invalid_argument);
  EXPECT_EQ(type.alignment(), 0);
  EXPECT_TRUE(type.variable_size());
  EXPECT_FALSE(type.is_primitive());
  legate::Type other(type);
  EXPECT_EQ(other, type);

  auto list_type = type.as_list_type();
  EXPECT_EQ(list_type.to_string(), to_string);
  EXPECT_EQ(list_type.element_type(), element_type);
}

void test_reduction_op(const legate::Type& type)
{
  type.record_reduction_operator(static_cast<int32_t>(legate::ReductionOpKind::ADD), GLOBAL_OP_ID);
  EXPECT_EQ(type.find_reduction_operator(static_cast<int32_t>(legate::ReductionOpKind::ADD)),
            GLOBAL_OP_ID);
  EXPECT_EQ(type.find_reduction_operator(legate::ReductionOpKind::ADD), GLOBAL_OP_ID);

  // repeat records for same type
  EXPECT_THROW(type.record_reduction_operator(static_cast<int32_t>(legate::ReductionOpKind::ADD),
                                              GLOBAL_OP_ID),
               std::invalid_argument);

  // reduction op doesn't exist
  EXPECT_THROW(type.find_reduction_operator(static_cast<int32_t>(legate::ReductionOpKind::SUB)),
               std::invalid_argument);
  EXPECT_THROW(type.find_reduction_operator(legate::ReductionOpKind::SUB), std::invalid_argument);
}

TEST(TypeUnit, PrimitiveType)
{
  test_primitive_type<bool>(legate::bool_(), "bool");
  test_primitive_type<int8_t>(legate::int8(), "int8");
  test_primitive_type<int16_t>(legate::int16(), "int16");
  test_primitive_type<int32_t>(legate::int32(), "int32");
  test_primitive_type<int64_t>(legate::int64(), "int64");
  test_primitive_type<uint8_t>(legate::uint8(), "uint8");
  test_primitive_type<uint16_t>(legate::uint16(), "uint16");
  test_primitive_type<uint32_t>(legate::uint32(), "uint32");
  test_primitive_type<uint64_t>(legate::uint64(), "uint64");
  test_primitive_type<__half>(legate::float16(), "float16");
  test_primitive_type<float>(legate::float32(), "float32");
  test_primitive_type<double>(legate::float64(), "float64");
  test_primitive_type<complex<float>>(legate::complex64(), "complex64");
  test_primitive_type<complex<double>>(legate::complex128(), "complex128");

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

  EXPECT_THROW(legate::primitive_type(legate::Type::Code::FIXED_ARRAY), std::invalid_argument);
  EXPECT_THROW(legate::primitive_type(legate::Type::Code::STRUCT), std::invalid_argument);
  EXPECT_THROW(legate::primitive_type(legate::Type::Code::STRING), std::invalid_argument);
  EXPECT_THROW(legate::primitive_type(legate::Type::Code::LIST), std::invalid_argument);
  EXPECT_THROW(legate::primitive_type(legate::Type::Code::INVALID), std::invalid_argument);
}

TEST(TypeUnit, StringType) { test_string_type(legate::string_type()); }

TEST(TypeUnit, FixedArrayType)
{
  // element type is a primitive type
  {
    uint32_t N            = 10;
    auto element_type     = legate::uint64();
    auto fixed_array_type = legate::fixed_array_type(element_type, N);
    test_fixed_array_type(fixed_array_type, element_type, N, "uint64[10]");
  }

  // element type is not a primitive type
  {
    uint32_t N            = 10;
    auto element_type     = legate::fixed_array_type(legate::uint16(), N);
    auto fixed_array_type = legate::fixed_array_type(element_type, N);
    test_fixed_array_type(fixed_array_type, element_type, N, "uint16[10][10]");
  }

  // N > 0xFFU
  {
    uint32_t N            = 256;
    auto element_type     = legate::float64();
    auto fixed_array_type = legate::fixed_array_type(element_type, N);
    test_fixed_array_type(fixed_array_type, element_type, N, "float64[256]");
  }

  // N = 0
  {
    EXPECT_THROW(legate::fixed_array_type(legate::int64(), 0), std::out_of_range);
  }

  // element type has variable size
  EXPECT_THROW(legate::fixed_array_type(legate::string_type(), 10), std::invalid_argument);

  // invalid casts
  EXPECT_THROW(legate::uint32().as_fixed_array_type(), std::invalid_argument);
}

TEST(TypeUnit, StructType)
{
  // aligned
  {
    auto type = legate::struct_type(true, legate::int16(), legate::bool_(), legate::float64());
    std::vector<legate::Type> field_types = {legate::int16(), legate::bool_(), legate::float64()};
    std::vector<uint32_t> offsets         = {0, 2, 8};
    // size: 16 = 8 (int16_t bool) + 8 (double)
    test_struct_type(
      type, true, 16, sizeof(double), "{int16:0,bool:2,float64:8}", field_types, offsets);
  }

  // aligned
  {
    std::vector<legate::Type> field_types = {legate::bool_(), legate::float64(), legate::int16()};
    auto type                             = legate::struct_type(field_types, true);
    std::vector<uint32_t> offsets         = {0, 8, 16};
    // size: 24 = 8 (bool) + 8 (double) + 8 (int16_t)
    test_struct_type(
      type, true, 24, sizeof(double), "{bool:0,float64:8,int16:16}", field_types, offsets);
  }

  // not aligned
  {
    auto type = legate::struct_type(false, legate::int16(), legate::bool_(), legate::float64());
    std::vector<legate::Type> field_types = {legate::int16(), legate::bool_(), legate::float64()};
    std::vector<uint32_t> offsets         = {0, 2, 3};
    auto size                             = sizeof(int16_t) + sizeof(bool) + sizeof(double);
    test_struct_type(type, false, size, 1, "{int16:0,bool:2,float64:3}", field_types, offsets);
  }

  // not aligned
  {
    std::vector<legate::Type> field_types = {legate::bool_(), legate::float64(), legate::int16()};
    auto type                             = legate::struct_type(field_types);
    std::vector<uint32_t> offsets         = {0, 1, 9};
    auto size                             = sizeof(int16_t) + sizeof(bool) + sizeof(double);
    test_struct_type(type, false, size, 1, "{bool:0,float64:1,int16:9}", field_types, offsets);
  }

  // field type has variable size
  EXPECT_THROW(legate::struct_type(true, legate::string_type(), legate::int16()),
               std::runtime_error);
  EXPECT_THROW(legate::struct_type(false, legate::string_type()), std::runtime_error);
  EXPECT_THROW(legate::struct_type(true), std::invalid_argument);
  EXPECT_THROW(legate::struct_type(false), std::invalid_argument);

  // invalid casts
  EXPECT_THROW(legate::uint32().as_struct_type(), std::invalid_argument);
}

TEST(TypeUnit, PointType)
{
  for (uint32_t idx = 1; idx <= LEGATE_MAX_DIM; ++idx) {
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
  EXPECT_THROW(legate::point_type(-1), std::out_of_range);
  EXPECT_THROW(legate::point_type(0), std::out_of_range);
  EXPECT_THROW(legate::point_type(LEGATE_MAX_DIM + 1), std::out_of_range);
}

TEST(TypeUnit, RectType)
{
  for (uint32_t idx = 1; idx <= LEGATE_MAX_DIM; ++idx) {
    auto type                             = legate::rect_type(idx);
    std::vector<legate::Type> field_types = {legate::point_type(idx), legate::point_type(idx)};
    std::vector<uint32_t> offsets         = {0, (uint32_t)(sizeof(uint64_t)) * idx};
    auto full_size                        = (field_types.size() * sizeof(uint64_t)) * idx;
    auto to_string = "{int64[" + std::to_string(idx) + "]:0,int64[" + std::to_string(idx) +
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
  EXPECT_THROW(legate::rect_type(-1), std::out_of_range);
  EXPECT_THROW(legate::rect_type(0), std::out_of_range);
  EXPECT_THROW(legate::rect_type(LEGATE_MAX_DIM + 1), std::out_of_range);
}

TEST(TypeUnit, ListType)
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
  EXPECT_THROW(legate::list_type(legate::string_type()), std::runtime_error);
  EXPECT_THROW(legate::list_type(legate::list_type(legate::uint32())), std::runtime_error);

  // invald casts
  EXPECT_THROW(legate::string_type().as_struct_type(), std::invalid_argument);
}

TEST(TypeUnit, Uid)
{
  // fixed array type
  for (uint32_t idx = 0; idx < PRIMITIVE_TYPE.size(); ++idx) {
    auto element_type     = PRIMITIVE_TYPE.at(idx);
    auto N                = idx + 1;
    auto fixed_array_type = legate::fixed_array_type(element_type, N);
    EXPECT_EQ(fixed_array_type.uid() & 0x00FF, static_cast<int32_t>(element_type.code()));
    EXPECT_EQ(fixed_array_type.uid() >> 8, N);
    EXPECT_EQ(fixed_array_type.as_fixed_array_type().num_elements(), N);

    EXPECT_NE(fixed_array_type.uid(), legate::string_type().uid());
    EXPECT_NE(fixed_array_type.uid(), static_cast<int32_t>(element_type.code()));

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
    auto dim                  = N % LEGATE_MAX_DIM + 1;
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
  EXPECT_EQ(legate::string_type().uid(), static_cast<int32_t>(legate::Type::Code::STRING));

  // struct type
  for (uint32_t idx = 0; idx < PRIMITIVE_TYPE.size(); ++idx) {
    auto element_type = PRIMITIVE_TYPE.at(idx);
    auto struct_type1 = legate::struct_type(true, element_type);
    auto struct_type2 = legate::struct_type(true, element_type);
    EXPECT_NE(struct_type1.uid(), struct_type2.uid());
    EXPECT_TRUE(struct_type1.uid() >= 0x10000);
    EXPECT_TRUE(struct_type2.uid() >= 0x10000);

    EXPECT_NE(struct_type1.uid(), legate::string_type().uid());
    EXPECT_NE(struct_type2.uid(), legate::string_type().uid());
    EXPECT_NE(struct_type1.uid(), static_cast<int32_t>(element_type.code()));
    EXPECT_NE(struct_type2.uid(), static_cast<int32_t>(element_type.code()));
  }

  // list type
  for (uint32_t idx = 0; idx < PRIMITIVE_TYPE.size(); ++idx) {
    auto element_type = PRIMITIVE_TYPE.at(idx);
    auto list_type1   = legate::list_type(element_type);
    auto list_type2   = legate::list_type(element_type);
    EXPECT_NE(list_type1.uid(), list_type2.uid());
    EXPECT_TRUE(list_type1.uid() >= 0x10000);
    EXPECT_TRUE(list_type2.uid() >= 0x10000);

    EXPECT_NE(list_type1.uid(), legate::string_type().uid());
    EXPECT_NE(list_type2.uid(), legate::string_type().uid());
    EXPECT_NE(list_type1.uid(), static_cast<int32_t>(element_type.code()));
    EXPECT_NE(list_type2.uid(), static_cast<int32_t>(element_type.code()));
  }
}

TEST(TypeUnit, ReductionOperator)
{
  test_reduction_op(legate::string_type());
  test_reduction_op(legate::fixed_array_type(legate::int64(), 10));
  test_reduction_op(legate::struct_type(true, legate::int64(), legate::bool_(), legate::float64()));
}

TEST(TypeUnit, TypeCodeOf)
{
  EXPECT_EQ(legate::legate_type_code_of<void>, legate::Type::Code::INVALID);
  EXPECT_EQ(legate::legate_type_code_of<bool>, legate::Type::Code::BOOL);
  EXPECT_EQ(legate::legate_type_code_of<int8_t>, legate::Type::Code::INT8);
  EXPECT_EQ(legate::legate_type_code_of<int16_t>, legate::Type::Code::INT16);
  EXPECT_EQ(legate::legate_type_code_of<int32_t>, legate::Type::Code::INT32);
  EXPECT_EQ(legate::legate_type_code_of<int64_t>, legate::Type::Code::INT64);
  EXPECT_EQ(legate::legate_type_code_of<uint8_t>, legate::Type::Code::UINT8);
  EXPECT_EQ(legate::legate_type_code_of<uint16_t>, legate::Type::Code::UINT16);
  EXPECT_EQ(legate::legate_type_code_of<uint32_t>, legate::Type::Code::UINT32);
  EXPECT_EQ(legate::legate_type_code_of<uint64_t>, legate::Type::Code::UINT64);
  EXPECT_EQ(legate::legate_type_code_of<__half>, legate::Type::Code::FLOAT16);
  EXPECT_EQ(legate::legate_type_code_of<float>, legate::Type::Code::FLOAT32);
  EXPECT_EQ(legate::legate_type_code_of<double>, legate::Type::Code::FLOAT64);
  EXPECT_EQ(legate::legate_type_code_of<complex<float>>, legate::Type::Code::COMPLEX64);
  EXPECT_EQ(legate::legate_type_code_of<complex<double>>, legate::Type::Code::COMPLEX128);
  EXPECT_EQ(legate::legate_type_code_of<std::string>, legate::Type::Code::STRING);
}

TEST(TypeUnit, TypeOf)
{
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::BOOL>, bool>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::INT8>, int8_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::INT16>, int16_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::INT32>, int32_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::INT64>, int64_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::UINT8>, uint8_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::UINT16>, uint16_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::UINT32>, uint32_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::UINT64>, uint64_t>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::FLOAT16>, __half>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::FLOAT32>, float>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::FLOAT64>, double>));
  EXPECT_TRUE(
    (std::is_same_v<legate::legate_type_of<legate::Type::Code::COMPLEX64>, complex<float>>));
  EXPECT_TRUE(
    (std::is_same_v<legate::legate_type_of<legate::Type::Code::COMPLEX128>, complex<double>>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::STRING>, std::string>));

  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::INVALID>, void>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::FIXED_ARRAY>, void>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::STRUCT>, void>));
  EXPECT_TRUE((std::is_same_v<legate::legate_type_of<legate::Type::Code::LIST>, void>));
}

TEST(TypeUnit, TypeUtils)
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

  EXPECT_FALSE(legate::is_integral<legate::Type::Code::INVALID>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_integral<legate::Type::Code::LIST>::value);

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

  EXPECT_FALSE(legate::is_signed<legate::Type::Code::INVALID>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_signed<legate::Type::Code::LIST>::value);

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

  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::INVALID>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_unsigned<legate::Type::Code::LIST>::value);

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

  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::INVALID>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_floating_point<legate::Type::Code::LIST>::value);

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

  EXPECT_FALSE(legate::is_complex<legate::Type::Code::INVALID>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::STRING>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::FIXED_ARRAY>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::STRUCT>::value);
  EXPECT_FALSE(legate::is_complex<legate::Type::Code::LIST>::value);

  // is_complex_type
  EXPECT_TRUE(legate::is_complex_type<complex<float>>::value);
  EXPECT_TRUE(legate::is_complex_type<complex<double>>::value);

  EXPECT_FALSE(legate::is_complex_type<bool>::value);
  EXPECT_FALSE(legate::is_complex_type<int8_t>::value);
  EXPECT_FALSE(legate::is_complex_type<int16_t>::value);
  EXPECT_FALSE(legate::is_complex_type<int32_t>::value);
  EXPECT_FALSE(legate::is_complex_type<int64_t>::value);
  EXPECT_FALSE(legate::is_complex_type<uint8_t>::value);
  EXPECT_FALSE(legate::is_complex_type<uint16_t>::value);
  EXPECT_FALSE(legate::is_complex_type<uint32_t>::value);
  EXPECT_FALSE(legate::is_complex_type<uint64_t>::value);
  EXPECT_FALSE(legate::is_complex_type<__half>::value);
  EXPECT_FALSE(legate::is_complex_type<float>::value);
  EXPECT_FALSE(legate::is_complex_type<double>::value);

  EXPECT_FALSE(legate::is_complex_type<void>::value);
  EXPECT_FALSE(legate::is_complex_type<std::string>::value);
}
}  // namespace type_test
