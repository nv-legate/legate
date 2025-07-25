/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace type_features_test {

namespace {

using TypeFeaturesUnit = DefaultFixture;

class UidTest : public TypeFeaturesUnit, public ::testing::WithParamInterface<legate::Type> {};

class UidFixedArrayTypeTest
  : public TypeFeaturesUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t>> {};

class ReductionOperatorTest
  : public TypeFeaturesUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, legate::ReductionOpKind>> {};

INSTANTIATE_TEST_SUITE_P(
  TypeFeaturesUnit,
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
                     ::testing::Range(1U, static_cast<std::uint32_t>(LEGATE_MAX_DIM))));

INSTANTIATE_TEST_SUITE_P(TypeFeaturesUnit,
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
  TypeFeaturesUnit,
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

}  // namespace

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
  constexpr auto big_size = 300;
  auto big_array_type     = legate::fixed_array_type(GetParam(), big_size);

  ASSERT_TRUE(big_array_type.uid() >= 0x10000);
}

TEST_F(TypeFeaturesUnit, StringTypeUid)
{
  ASSERT_EQ(legate::string_type().uid(), static_cast<std::int32_t>(legate::Type::Code::STRING));
}

TEST_P(UidTest, StructType)
{
  const auto& element_type = GetParam();
  auto struct_type1        = legate::struct_type(true, element_type);
  auto struct_type2        = legate::struct_type(true, element_type);

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
  const auto& element_type = GetParam();
  auto list_type1          = legate::list_type(element_type);
  auto list_type2          = legate::list_type(element_type);

  ASSERT_NE(list_type1.uid(), list_type2.uid());
  ASSERT_TRUE(list_type1.uid() >= 0x10000);
  ASSERT_TRUE(list_type2.uid() >= 0x10000);

  ASSERT_NE(list_type1.uid(), legate::string_type().uid());
  ASSERT_NE(list_type2.uid(), legate::string_type().uid());
  ASSERT_NE(list_type1.uid(), static_cast<std::int32_t>(element_type.code()));
  ASSERT_NE(list_type2.uid(), static_cast<std::int32_t>(element_type.code()));
}

TEST_F(TypeFeaturesUnit, BinaryTypeUid)
{
  constexpr auto size   = 678;
  auto binary_type      = legate::binary_type(size);
  auto same_binary_type = legate::binary_type(size);
  auto diff_binary_type = legate::binary_type(size + 1);

  ASSERT_EQ(binary_type.uid(), same_binary_type.uid());
  ASSERT_NE(binary_type.uid(), diff_binary_type.uid());
}

TEST_P(ReductionOperatorTest, Record)
{
  constexpr auto GLOBAL_OP_ID = legate::GlobalRedopID{0x1F};
  const auto& param           = GetParam();
  const auto type             = std::get<0>(param);
  const auto op_kind          = std::get<1>(param);

  type.record_reduction_operator(static_cast<std::int32_t>(op_kind), GLOBAL_OP_ID);
  ASSERT_EQ(type.find_reduction_operator(static_cast<std::int32_t>(op_kind)), GLOBAL_OP_ID);
  ASSERT_EQ(type.find_reduction_operator(op_kind), GLOBAL_OP_ID);

  // repeat records for same type
  ASSERT_THAT(
    [&]() { type.record_reduction_operator(static_cast<std::int32_t>(op_kind), GLOBAL_OP_ID); },
    testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("already exists for type")));
}

}  // namespace type_features_test
