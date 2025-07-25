/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace list_type_test {

namespace {

using ListTypeUnit = DefaultFixture;

class ListTypeTest : public ListTypeUnit,
                     public ::testing::WithParamInterface<std::tuple<legate::Type, std::string>> {};

INSTANTIATE_TEST_SUITE_P(
  ListTypeUnit,
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

}  // namespace

TEST_P(ListTypeTest, Basic)
{
  const auto [element_type, to_string] = GetParam();
  const auto type                      = legate::list_type(element_type);

  ASSERT_EQ(type.code(), legate::Type::Code::LIST);
  ASSERT_THAT([&]() { static_cast<void>(type.size()); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Size of a variable size type is undefined")));
  ASSERT_EQ(type.alignment(), 0);
  ASSERT_TRUE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());

  // Note: aim to test the copy initialization of Type
  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);

  auto list_type = type.as_list_type();

  ASSERT_EQ(list_type.to_string(), to_string);
  ASSERT_EQ(list_type.element_type(), element_type);
}

TEST_F(ListTypeUnit, Equal)
{
  auto type = legate::list_type(legate::int16());

  ASSERT_TRUE(type == type);
}

TEST_F(ListTypeUnit, NotEqual)
{
  auto type1 = legate::list_type(legate::int16());
  auto type2 = legate::list_type(legate::int32());

  ASSERT_FALSE(type1 == type2);
}

TEST_F(ListTypeUnit, ListTypeBadType)
{
  ASSERT_THAT([&]() { static_cast<void>(legate::list_type(legate::string_type())); },
              testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Nested variable size types are not implemented yet")));
  ASSERT_THAT([&]() { static_cast<void>(legate::list_type(legate::list_type(legate::uint32()))); },
              testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Nested variable size types are not implemented yet")));
}

TEST_F(ListTypeUnit, ListTypeBadCast)
{
  ASSERT_THAT(
    [&]() { static_cast<void>(legate::string_type().as_list_type()); },
    testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("Type is not a list type")));
}

}  // namespace list_type_test
