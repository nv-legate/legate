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

#include <legate.h>

#include <gtest/gtest.h>

#include <unit/logical_array/utils.h>
#include <utilities/utilities.h>

namespace logical_array_create_test {

namespace {

using LogicalArrayCreateUnit = DefaultFixture;

constexpr auto BOUND_DIM                = 4;
constexpr auto VARILABLE_TYPE_BOUND_DIM = 1;
constexpr auto UNBOUND_DIM              = 1;
constexpr auto NUM_CHILDREN             = 2;

const legate::Shape& bound_shape_multi_dim()
{
  static const auto shape = legate::Shape{1, 2, 3, 4};

  return shape;
}

const legate::Shape& bound_shape_single_dim()
{
  static const auto shape = legate::Shape{10};

  return shape;
}

class NullableTest : public LogicalArrayCreateUnit, public ::testing::WithParamInterface<bool> {};

class CreateInvalidArrayTest
  : public LogicalArrayCreateUnit,
    public ::testing::WithParamInterface<std::tuple<const legate::LogicalArray /* descriptor */,
                                                    const legate::LogicalArray /* vardata */>> {};

INSTANTIATE_TEST_SUITE_P(LogicalArrayCreateUnit, NullableTest, ::testing::Values(true, false));

std::vector<std::tuple<const legate::LogicalArray, const legate::LogicalArray>>
generate_invalid_arrays()
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
    runtime->create_array(legate::Shape{1}, legate::int8(), true /*nullable*/);

  return {
    std::make_tuple(arr_unbound_rect1, arr_int8) /* descriptor: unbound sub-array */,
    std::make_tuple(arr_rect1, arr_unbound_int8) /* vardata: unbound sub-array */,
    std::make_tuple(arr_4d_rect1, arr_int8) /* descriptor: multi-dimensional sub-array */,
    std::make_tuple(arr_rect1, arr_4d_int8) /* vardata: multi-dimensional sub-array */,
    std::make_tuple(arr_int64, arr_int8) /* incorrect descriptor type */,
    std::make_tuple(arr_rect1, arr_nullable_int8) /* nullable vardata */
  };
}

INSTANTIATE_TEST_SUITE_P(LogicalArrayCreateUnit,
                         CreateInvalidArrayTest,
                         ::testing::ValuesIn(generate_invalid_arrays()));

}  // namespace

TEST_P(NullableTest, CreateBoundPrimitiveArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int64();
  auto array = logical_array_util_test::create_array_with_type(primitive_type, true, nullable);

  ASSERT_FALSE(array.unbound());
  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_multi_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_multi_dim().volume());
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), 0);
  ASSERT_FALSE(array.nested());

  auto store = array.data();

  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), BOUND_DIM);
  ASSERT_EQ(store.extents(), bound_shape_multi_dim().extents());
  ASSERT_EQ(store.volume(), bound_shape_multi_dim().volume());
  ASSERT_EQ(store.type(), primitive_type);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, CreateUnboundPrimitiveArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int32();
  auto array = logical_array_util_test::create_array_with_type(primitive_type, false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), 0);
  ASSERT_FALSE(array.nested());

  auto store = array.data();

  ASSERT_TRUE(store.unbound());
  ASSERT_EQ(store.dim(), UNBOUND_DIM);
  ASSERT_THROW(static_cast<void>(store.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(store.volume()), std::invalid_argument);
  ASSERT_EQ(store.type(), primitive_type);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, CreateBoundListArray)
{
  const auto nullable = GetParam();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto array          = logical_array_util_test::create_array_with_type(arr_type, true, nullable);

  // List arrays are unbound even with the fixed extents
  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  ASSERT_EQ(array.type(), arr_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto list_array = array.as_list_array();

  // Sub-arrays of list arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(list_array.descriptor()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.vardata()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateUnboundListArray)
{
  const auto nullable = GetParam();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto array          = logical_array_util_test::create_array_with_type(arr_type, false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
  ASSERT_EQ(array.type(), arr_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto list_array = array.as_list_array();

  // Sub-arrays of list arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(list_array.descriptor()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.vardata()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateListArray)
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
  // compare code here since UID of ListType differ between differnt objects
  ASSERT_EQ(array.type().code(), legate::Type::Code::LIST);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);

  auto list_array = array.as_list_array();
  // Sub-arrays can be accessed
  static_cast<void>(list_array.descriptor());
  static_cast<void>(list_array.vardata());
}

TEST_P(NullableTest, CreateBoundStringArray)
{
  const auto nullable = GetParam();
  auto str_type       = legate::string_type();
  auto array          = logical_array_util_test::create_array_with_type(str_type, true, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  ASSERT_EQ(array.type(), str_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto string_array = array.as_string_array();

  // Sub-arrays of string arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(string_array.chars()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.offsets()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateUnboundStringArray)
{
  const auto nullable = GetParam();
  auto str_type       = legate::string_type();
  auto array          = logical_array_util_test::create_array_with_type(str_type, false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
  ASSERT_EQ(array.type(), str_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto string_array = array.as_string_array();

  // Sub-arrays of string arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(string_array.chars()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.offsets()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateBoundStructArray)
{
  const auto nullable = GetParam();
  const auto& st_type = logical_array_util_test::struct_type();
  auto num_fields     = st_type.num_fields();
  auto array          = logical_array_util_test::create_array_with_type(st_type, true, nullable);

  ASSERT_FALSE(array.unbound());
  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_multi_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_multi_dim().volume());
  ASSERT_EQ(array.type(), st_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), num_fields);
  ASSERT_TRUE(array.nested());

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(num_fields)), std::out_of_range);
  for (std::uint32_t idx = 0; idx < num_fields; ++idx) {
    auto field_type     = st_type.field_type(idx);
    auto field_subarray = array.child(idx);

    ASSERT_EQ(field_subarray.unbound(), array.unbound());
    ASSERT_EQ(field_subarray.dim(), array.dim());
    ASSERT_EQ(field_subarray.extents(), array.extents());
    ASSERT_EQ(field_subarray.volume(), array.volume());
    ASSERT_EQ(field_subarray.type(), field_type);
    // There'd be only one null mask for the whole struct array
    ASSERT_EQ(field_subarray.nullable(), false);
    ASSERT_THROW(static_cast<void>(field_subarray.null_mask()), std::invalid_argument);
    ASSERT_EQ(field_subarray.num_children(), 0);
  }
}

TEST_P(NullableTest, CreateUnboundStructArray)
{
  const auto nullable = GetParam();
  const auto& st_type = logical_array_util_test::struct_type();
  auto num_fields     = st_type.num_fields();
  auto array          = logical_array_util_test::create_array_with_type(st_type, false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_EQ(array.type(), st_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), num_fields);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, CreateBoundLike)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto source  = logical_array_util_test::create_array_with_type(legate::int64(), true, nullable);
  auto target1 = runtime->create_array_like(source);

  ASSERT_EQ(source.dim(), target1.dim());
  ASSERT_EQ(source.type(), target1.type());
  ASSERT_EQ(source.extents(), target1.extents());
  ASSERT_EQ(source.volume(), target1.volume());
  ASSERT_EQ(source.unbound(), target1.unbound());
  ASSERT_EQ(source.nullable(), target1.nullable());

  auto target2 = runtime->create_array_like(source, legate::float64());

  ASSERT_EQ(source.dim(), target2.dim());
  ASSERT_EQ(target2.type(), legate::float64());
  ASSERT_EQ(source.extents(), target2.extents());
  ASSERT_EQ(source.volume(), target2.volume());
  ASSERT_EQ(source.unbound(), target2.unbound());
  ASSERT_EQ(source.nullable(), target2.nullable());
}

TEST_P(NullableTest, CreateUnboundLike)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto source  = logical_array_util_test::create_array_with_type(legate::int64(), false, nullable);
  auto target1 = runtime->create_array_like(source);

  ASSERT_EQ(source.dim(), target1.dim());
  ASSERT_EQ(source.type(), target1.type());
  ASSERT_THROW(static_cast<void>(target1.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(target1.volume()), std::invalid_argument);
  ASSERT_EQ(source.unbound(), target1.unbound());
  ASSERT_EQ(source.nullable(), target1.nullable());

  auto target2 = runtime->create_array_like(source, legate::float64());

  ASSERT_EQ(source.dim(), target2.dim());
  ASSERT_EQ(target2.type(), legate::float64());
  ASSERT_THROW(static_cast<void>(target2.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(target2.volume()), std::invalid_argument);
  ASSERT_EQ(source.unbound(), target2.unbound());
  ASSERT_EQ(source.nullable(), target2.nullable());
}

TEST_P(NullableTest, InvalidCastBoundArray)
{
  auto bound_array =
    logical_array_util_test::create_array_with_type(legate::int64(), true, GetParam());

  ASSERT_THROW(static_cast<void>(bound_array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(bound_array.as_string_array()), std::invalid_argument);
}

TEST_P(NullableTest, InvalidCastUnboundArray)
{
  auto unbound_array =
    logical_array_util_test::create_array_with_type(legate::int64(), false, GetParam());

  ASSERT_THROW(static_cast<void>(unbound_array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(unbound_array.as_string_array()), std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidCreateListArray)
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional list arrays are not allowed
  ASSERT_THROW(static_cast<void>(runtime->create_array(bound_shape_multi_dim(),
                                                       legate::list_type(legate::int64()))),
               std::invalid_argument);
  ASSERT_THROW(
    static_cast<void>(runtime->create_array(legate::list_type(legate::int64()), BOUND_DIM)),
    std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidCreateStringArray)
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional string arrays are not allowed
  ASSERT_THROW(
    static_cast<void>(runtime->create_array(bound_shape_multi_dim(), legate::string_type())),
    std::invalid_argument);
  ASSERT_THROW(static_cast<void>(runtime->create_array(legate::string_type(), BOUND_DIM)),
               std::invalid_argument);
}

TEST_P(CreateInvalidArrayTest, StringArray)
{
  const auto [descriptor, vardata] = GetParam();
  auto runtime                     = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->create_string_array(descriptor, vardata)),
               std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidStringArray)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{1}, legate::int64());

  // incorrect vardata type
  ASSERT_THROW(static_cast<void>(runtime->create_string_array(arr_rect1, arr_int64)),
               std::invalid_argument);
}

TEST_P(CreateInvalidArrayTest, ListArray)
{
  const auto [descriptor, vardata] = GetParam();
  auto runtime                     = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->create_list_array(descriptor, vardata)),
               std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidListArray)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{1}, legate::int64());

  // incorrect type
  ASSERT_THROW(static_cast<void>(runtime->create_list_array(arr_rect1, arr_int64, legate::int64())),
               std::invalid_argument);

  // incorrect vardata type
  ASSERT_THROW(static_cast<void>(runtime->create_list_array(
                 arr_rect1, arr_int64, legate::list_type(legate::int8()))),
               std::invalid_argument);
}

TEST_P(NullableTest, PhsicalArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int64();
  auto logical_array =
    logical_array_util_test::create_array_with_type(primitive_type, true, nullable);
  auto array = logical_array.get_physical_array();

  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_FALSE(array.nested());

  auto shape = legate::Rect<BOUND_DIM>({0, 0, 0, 0}, {0, 1, 2, 3});

  ASSERT_EQ(array.shape<BOUND_DIM>(), shape);
  ASSERT_EQ((array.domain().bounds<BOUND_DIM, std::int64_t>()), shape);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.shape<BOUND_DIM>(), array.shape<BOUND_DIM>());
    ASSERT_EQ(null_mask.domain(), array.domain());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

}  // namespace logical_array_create_test
