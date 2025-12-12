/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace struct_type_test {

namespace {

using StructTypeUnit = DefaultFixture;

class StructTypeTest
  : public StructTypeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type,
                                                    std::vector<legate::Type>,
                                                    bool,
                                                    std::uint32_t /* total_size */,
                                                    std::uint32_t /* alignment */,
                                                    std::string>> {};

class DimTest : public StructTypeUnit, public ::testing::WithParamInterface<std::uint32_t> {};

class RectTypeTest : public DimTest {};

class RectTypeNegativeDimTest : public DimTest {};

class RectTypeDimMismatchTest : public DimTest {};

class RectTypeMismatchTest
  : public StructTypeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t>> {};

INSTANTIATE_TEST_SUITE_P(
  StructTypeUnit,
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
        std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()})),
      std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()}),
      true,
      24,
      sizeof(double),
      "{bool:0,float64:8,int16:16}"),
    std::make_tuple(
      legate::struct_type(false, legate::int16(), legate::bool_(), legate::float64()),
      std::vector<legate::Type>({legate::int16(), legate::bool_(), legate::float64()}),
      false,
      sizeof(std::int16_t) + sizeof(bool) + sizeof(double),
      sizeof(bool),
      "{int16:0,bool:2,float64:3}"),
    std::make_tuple(
      legate::struct_type(
        std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()}), false),
      std::vector<legate::Type>({legate::bool_(), legate::float64(), legate::int16()}),
      false,
      sizeof(std::int16_t) + sizeof(bool) + sizeof(double),
      sizeof(bool),
      "{bool:0,float64:1,int16:9}")));

INSTANTIATE_TEST_SUITE_P(StructTypeUnit,
                         RectTypeTest,
                         ::testing::Range(1U, static_cast<std::uint32_t>(LEGATE_MAX_DIM)));

INSTANTIATE_TEST_SUITE_P(StructTypeUnit,
                         RectTypeNegativeDimTest,
                         ::testing::Values(-1, 0, LEGATE_MAX_DIM + 1));

INSTANTIATE_TEST_SUITE_P(StructTypeUnit, RectTypeDimMismatchTest, ::testing::Values(-1, 0, 2));

INSTANTIATE_TEST_SUITE_P(
  StructTypeUnit,
  RectTypeMismatchTest,
  ::testing::Values(std::make_tuple(legate::point_type(1), 1),
                    std::make_tuple(legate::fixed_array_type(legate::int64(), 2), 2),
                    std::make_tuple(legate::int64(), LEGATE_MAX_DIM)));

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

  // Note: aim to test the copy initialization of Type
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

}  // namespace

TEST_P(StructTypeTest, Basic)
{
  const auto [type, field_types, align, total_size, alignment, to_string] = GetParam();

  test_struct_type(type, align, total_size, alignment, to_string, field_types);
}

TEST_F(StructTypeUnit, Equal)
{
  auto type =
    legate::struct_type(/*align=*/true, legate::int16(), legate::bool_(), legate::float64());

  ASSERT_TRUE(type == type);
}

TEST_F(StructTypeUnit, NotEqual)
{
  auto type1 = legate::struct_type(/*align=*/true, legate::int16());
  auto type2 = legate::int16();

  ASSERT_FALSE(type1 == type2);
}

TEST_F(StructTypeUnit, StructTypeBadFields)
{
  ASSERT_THAT(
    [&]() { static_cast<void>(legate::struct_type(true, legate::string_type(), legate::int16())); },
    testing::ThrowsMessage<std::runtime_error>(
      ::testing::HasSubstr("Struct types can't have a variable size field")));
  ASSERT_THAT([&]() { static_cast<void>(legate::struct_type(false, legate::string_type())); },
              testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Struct types can't have a variable size field")));
}

TEST_F(StructTypeUnit, StructTypeEmptyFields)
{
  ASSERT_THAT([&]() { static_cast<void>(legate::struct_type(true)); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Struct types must have at least one field")));
  ASSERT_THAT([&]() { static_cast<void>(legate::struct_type(false)); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Struct types must have at least one field")));
}

TEST_F(StructTypeUnit, StructTypeBadCast)
{
  ASSERT_THAT([&]() { static_cast<void>(legate::uint32().as_struct_type()); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Type is not a struct type")));
}

TEST_P(RectTypeTest, Basic)
{
  const auto dim                              = GetParam();
  const auto type                             = legate::rect_type(dim);
  const std::vector<legate::Type> field_types = {legate::point_type(dim), legate::point_type(dim)};
  const auto full_size                        = (field_types.size() * sizeof(std::uint64_t)) * dim;
  const auto to_string =
    fmt::format("{{int64[{}]:0,int64[{}]:{}}}", dim, dim, dim * sizeof(std::int64_t));

  test_struct_type(
    type, /*aligned=*/true, full_size, sizeof(std::uint64_t), to_string, field_types);
  ASSERT_TRUE(legate::is_rect_type(type, dim));
  ASSERT_TRUE(legate::is_rect_type(type));
  ASSERT_EQ(legate::ndim_rect_type(type), dim);
}

TEST_P(RectTypeDimMismatchTest, Basic)
{
  ASSERT_FALSE(legate::is_rect_type(legate::rect_type(1), GetParam()));
}

TEST_P(RectTypeMismatchTest, Basic)
{
  const auto& param = GetParam();
  const auto type   = std::get<0>(param);
  const auto size   = std::get<1>(param);

  ASSERT_FALSE(legate::is_rect_type(type, size));
  ASSERT_FALSE(legate::is_rect_type(type));
  ASSERT_THAT([&]() { static_cast<void>(legate::ndim_rect_type(type)); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Expected a rect type but got")));
}

TEST_P(RectTypeNegativeDimTest, RectType)
{
  ASSERT_THAT([&]() { static_cast<void>(legate::rect_type(GetParam())); },
              testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("is not a supported number of dimensions")));
}

}  // namespace struct_type_test
