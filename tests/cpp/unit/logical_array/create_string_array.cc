/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/logical_array/utils.h>

namespace logical_array_create_test {

using logical_array_util_test::BOUND_DIM;
using logical_array_util_test::bound_shape_multi_dim;
using logical_array_util_test::bound_shape_single_dim;
using logical_array_util_test::NUM_CHILDREN;
using logical_array_util_test::UNBOUND_DIM;
using logical_array_util_test::VARILABLE_TYPE_BOUND_DIM;

namespace {

class StringArrayCreateUnit : public DefaultFixture {};

class StringNullableTest : public StringArrayCreateUnit,
                           public ::testing::WithParamInterface<bool> {};

class CreateInvalidStringArrayTest
  : public StringArrayCreateUnit,
    public ::testing::WithParamInterface<std::tuple<const legate::LogicalArray /* descriptor */,
                                                    const legate::LogicalArray /* vardata */,
                                                    std::string /* expected_error */>> {};

std::vector<std::tuple<const legate::LogicalArray, const legate::LogicalArray, std::string>>
generate_invalid_string_arrays()
{
  legate::start();
  auto runtime                 = legate::Runtime::get_runtime();
  const auto arr_unbound_rect1 = runtime->create_array(legate::rect_type(UNBOUND_DIM));
  const auto arr_unbound_int8  = runtime->create_array(legate::int8());
  const auto arr_rect1         = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int8          = runtime->create_array(legate::Shape{1}, legate::int8());
  const auto arr_4d_rect1 =
    runtime->create_array(bound_shape_multi_dim(), legate::rect_type(BOUND_DIM));
  const auto arr_4d_int8 = runtime->create_array(bound_shape_multi_dim(), legate::int8());
  const auto arr_nullable_int8 =
    runtime->create_array(legate::Shape{1}, legate::int8(), /*nullable=*/true);

  return {
    {arr_unbound_rect1, arr_int8, "Descriptor and vardata should not be unbound"},
    {arr_rect1, arr_unbound_int8, "Descriptor and vardata should not be unbound"},
    {arr_4d_rect1, arr_int8, "Descriptor and vardata should be 1D"},
    {arr_rect1, arr_4d_int8, "Descriptor and vardata should be 1D"},
    {arr_rect1, arr_nullable_int8, "Vardata should not be nullable"},
  };
}

INSTANTIATE_TEST_SUITE_P(StringArrayCreateUnit, StringNullableTest, ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(StringArrayCreateUnit,
                         CreateInvalidStringArrayTest,
                         ::testing::ValuesIn(generate_invalid_string_arrays()));

}  // namespace

TEST_P(StringNullableTest, Bound)
{
  const auto nullable = GetParam();
  auto str_type       = legate::string_type();
  auto array = logical_array_util_test::create_array_with_type(str_type, /*bound=*/true, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  ASSERT_EQ(array.type(), str_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THAT([&] { static_cast<void>(array.data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&] { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }

  auto string_array = array.as_string_array();

  // Sub-arrays of string arrays cannot be retrieved until initialized (string arrays are always
  // unbound)
  ASSERT_THAT([&] { static_cast<void>(string_array.chars()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(string_array.offsets()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(string_array.child(NUM_CHILDREN)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
}

TEST_P(StringNullableTest, Unbound)
{
  const auto nullable = GetParam();
  auto str_type       = legate::string_type();
  auto array = logical_array_util_test::create_array_with_type(str_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_THAT([&] { static_cast<void>(array.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(array.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_EQ(array.type(), str_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THAT([&] { static_cast<void>(array.data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THAT([&] { static_cast<void>(null_mask.extents()); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                  ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&] { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }

  auto string_array = array.as_string_array();

  // Sub-arrays of unbound string arrays cannot be retrieved until initialized
  ASSERT_THAT([&] { static_cast<void>(string_array.chars()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(string_array.offsets()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(string_array.child(NUM_CHILDREN)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
}

TEST_F(StringArrayCreateUnit, InvalidCreation)
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional string arrays are not allowed
  ASSERT_THAT(
    [&] {
      static_cast<void>(runtime->create_array(bound_shape_multi_dim(), legate::string_type()));
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("List/string arrays can only have 1D shapes")));
  ASSERT_THAT([&] { static_cast<void>(runtime->create_array(legate::string_type(), BOUND_DIM)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("List/string arrays can only have 1D shapes")));
}

TEST_P(CreateInvalidStringArrayTest, InvalidCreation)
{
  const auto& param          = GetParam();
  const auto& descriptor     = std::get<0>(param);
  const auto& vardata        = std::get<1>(param);
  const auto& expected_error = std::get<2>(param);
  auto runtime               = legate::Runtime::get_runtime();

  ASSERT_THAT(
    [&] { static_cast<void>(runtime->create_string_array(descriptor, vardata)); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(expected_error)));
}

TEST_F(StringArrayCreateUnit, InvalidVardataType)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{1}, legate::int64());

  // incorrect vardata type (string arrays require int8 vardata)
  ASSERT_THAT([&] { static_cast<void>(runtime->create_string_array(arr_rect1, arr_int64)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Expected a vardata array of type")));
}

}  // namespace logical_array_create_test
