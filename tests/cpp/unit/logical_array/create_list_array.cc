/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_arrays/list_logical_array.h>

#include <unit/logical_array/utils.h>

namespace logical_array_create_test {

using logical_array_util_test::BOUND_DIM;
using logical_array_util_test::bound_shape_multi_dim;
using logical_array_util_test::bound_shape_single_dim;
using logical_array_util_test::NUM_CHILDREN;
using logical_array_util_test::UNBOUND_DIM;
using logical_array_util_test::VARILABLE_TYPE_BOUND_DIM;

namespace {

class ListArrayCreateUnit : public DefaultFixture {};

class ListNullableTest : public ListArrayCreateUnit, public ::testing::WithParamInterface<bool> {};

class CreateInvalidListArrayTest
  : public ListArrayCreateUnit,
    public ::testing::WithParamInterface<std::tuple<const legate::LogicalArray /* descriptor */,
                                                    const legate::LogicalArray /* vardata */,
                                                    std::string /* expected_error */>> {};

std::vector<std::tuple<const legate::LogicalArray, const legate::LogicalArray, std::string>>
generate_invalid_list_arrays()
{
  legate::start();
  auto runtime                 = legate::Runtime::get_runtime();
  const auto arr_unbound_rect1 = runtime->create_array(legate::rect_type(UNBOUND_DIM));
  const auto arr_unbound_int8  = runtime->create_array(legate::int8());
  const auto arr_rect1         = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int8          = runtime->create_array(legate::Shape{1}, legate::int8());
  const auto arr_int64         = runtime->create_array(legate::Shape{1}, legate::int64());
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
    {arr_int64, arr_int8, "Descriptor array does not have a 1D rect type"},
    {arr_rect1, arr_nullable_int8, "Vardata should not be nullable"},
  };
}

INSTANTIATE_TEST_SUITE_P(ListArrayCreateUnit, ListNullableTest, ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(ListArrayCreateUnit,
                         CreateInvalidListArrayTest,
                         ::testing::ValuesIn(generate_invalid_list_arrays()));

}  // namespace

TEST_P(ListNullableTest, Bound)
{
  const auto nullable = GetParam();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto array = logical_array_util_test::create_array_with_type(arr_type, /*bound=*/true, nullable);

  // List arrays are unbound even with the fixed extents
  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  ASSERT_EQ(array.type(), arr_type);
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

  auto list_array = array.as_list_array();

  // Sub-arrays of list arrays cannot be retrieved until initialized (list arrays are always
  // unbound)
  ASSERT_THAT([&] { static_cast<void>(list_array.descriptor()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(list_array.vardata()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(list_array.child(NUM_CHILDREN)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
}

TEST_P(ListNullableTest, Unbound)
{
  const auto nullable = GetParam();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto array = logical_array_util_test::create_array_with_type(arr_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_THAT([&] { static_cast<void>(array.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(array.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_EQ(array.type(), arr_type);
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

  auto list_array = array.as_list_array();

  // Sub-arrays of unbound list arrays cannot be retrieved until initialized
  ASSERT_THAT([&] { static_cast<void>(list_array.descriptor()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(list_array.vardata()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
  ASSERT_THAT([&] { static_cast<void>(list_array.child(NUM_CHILDREN)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
}

TEST_P(ListNullableTest, Creation)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto type           = legate::list_type(legate::int64());
  auto descriptor     = runtime->create_array(
    bound_shape_single_dim(), legate::rect_type(VARILABLE_TYPE_BOUND_DIM), nullable);
  auto vardata = runtime->create_array(bound_shape_single_dim(), legate::int64());
  auto array   = runtime->create_list_array(descriptor, vardata, type);

  ASSERT_FALSE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  // compare code here since UID of ListType differ between different objects
  ASSERT_EQ(array.type().code(), legate::Type::Code::LIST);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);

  auto list_array = array.as_list_array();
  // Sub-arrays can be accessed
  ASSERT_NO_THROW(static_cast<void>(list_array.descriptor()));
  ASSERT_NO_THROW(static_cast<void>(list_array.vardata()));
  for (std::uint32_t idx = 0; idx < NUM_CHILDREN; ++idx) {
    ASSERT_NO_THROW(static_cast<void>(list_array.child(idx)));
  }
  ASSERT_THAT([&] { static_cast<void>(list_array.child(NUM_CHILDREN)); },
              ::testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("List array does not have child 2")));
}

TEST_P(ListNullableTest, PhysicalArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto type           = legate::list_type(legate::int64());
  auto descriptor     = runtime->create_array(
    bound_shape_single_dim(), legate::rect_type(VARILABLE_TYPE_BOUND_DIM), nullable);
  auto vardata = runtime->create_array(bound_shape_single_dim(), legate::int64());
  auto array   = runtime->create_list_array(descriptor, vardata, type);

  auto list_array          = array.as_list_array();
  auto physical_array      = list_array.get_physical_array();
  auto list_physical_array = physical_array.as_list_array();

  ASSERT_NO_THROW(static_cast<void>(list_physical_array.descriptor()));
  ASSERT_NO_THROW(static_cast<void>(list_physical_array.vardata()));
  ASSERT_EQ(list_physical_array.nullable(), nullable);
  ASSERT_EQ(list_physical_array.type(), type);
  ASSERT_EQ(list_physical_array.dim(), VARILABLE_TYPE_BOUND_DIM);
}

TEST_F(ListArrayCreateUnit, InvalidCreation)
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional list arrays are not allowed
  ASSERT_THAT(
    [&] {
      static_cast<void>(
        runtime->create_array(bound_shape_multi_dim(), legate::list_type(legate::int64())));
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("List/string arrays can only have 1D shapes")));
  ASSERT_THAT(
    [&] {
      static_cast<void>(runtime->create_array(legate::list_type(legate::int64()), BOUND_DIM));
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("List/string arrays can only have 1D shapes")));
}

TEST_P(CreateInvalidListArrayTest, InvalidDescriptorOrVardata)
{
  const auto& param          = GetParam();
  const auto& descriptor     = std::get<0>(param);
  const auto& vardata        = std::get<1>(param);
  const auto& expected_error = std::get<2>(param);
  auto runtime               = legate::Runtime::get_runtime();

  ASSERT_THAT(
    [&] { static_cast<void>(runtime->create_list_array(descriptor, vardata)); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(expected_error)));
}

TEST_F(ListArrayCreateUnit, InvalidVardataType)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{1}, legate::int64());

  // incorrect type (not a list type)
  ASSERT_THAT(
    [&] { static_cast<void>(runtime->create_list_array(arr_rect1, arr_int64, legate::int64())); },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Expected a list type but got")));

  // incorrect vardata type (list<int8> but vardata is int64)
  ASSERT_THAT(
    [&] {
      static_cast<void>(
        runtime->create_list_array(arr_rect1, arr_int64, legate::list_type(legate::int8())));
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Expected a vardata array of type")));
}

TEST_F(ListArrayCreateUnit, InvalidType)
{
  // Create a non-LIST/STRING type (INT32)
  auto invalid_type = legate::int32().impl();

  // Descriptor and vardata can be nullptr for this test since we expect
  // the type check to fail before they are used
  ASSERT_THAT(
    [&] { static_cast<void>(legate::detail::ListLogicalArray{invalid_type, nullptr, nullptr}); },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("expected list or string type")));
}

TEST_F(ListArrayCreateUnit, Broadcast)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto list_array = runtime->create_array(legate::Shape{2}, legate::list_type(legate::int64()));

  ASSERT_THAT([&] { static_cast<void>(list_array.broadcast(0, 10)); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("List array does not support store transformations")));
}

}  // namespace logical_array_create_test
