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

namespace logical_array_test {

using LogicalArrayUnit = DefaultFixture;

void test_primitive_array(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // Bound
  {
    auto array = runtime->create_array({4, 4}, legate::int64(), nullable);
    EXPECT_FALSE(array.unbound());
    EXPECT_EQ(array.dim(), 2);
    EXPECT_EQ(array.extents().data(), (std::vector<std::uint64_t>{4, 4}));
    EXPECT_EQ(array.volume(), 16);
    EXPECT_EQ(array.type(), legate::int64());
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 0);
    EXPECT_FALSE(array.nested());

    auto store = array.data();
    EXPECT_FALSE(store.unbound());
    EXPECT_EQ(store.dim(), 2);
    EXPECT_EQ(store.extents().data(), (std::vector<std::uint64_t>{4, 4}));
    EXPECT_EQ(store.volume(), 16);
    EXPECT_EQ(store.type(), legate::int64());

    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_EQ(null_mask.extents(), array.extents());
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }
    EXPECT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  }
  // Unbound
  {
    auto array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 3);
    EXPECT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
    EXPECT_EQ(array.type(), legate::int64());
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 0);
    EXPECT_FALSE(array.nested());

    auto store = array.data();
    EXPECT_TRUE(store.unbound());
    EXPECT_EQ(store.dim(), 3);
    EXPECT_THROW(static_cast<void>(store.extents()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.volume()), std::invalid_argument);
    EXPECT_EQ(store.type(), legate::int64());

    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }
    EXPECT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  }
}

void test_list_array(bool nullable)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto arr_type = legate::list_type(legate::int64()).as_list_type();
  // Bound descriptor
  {
    auto array = runtime->create_array({7}, arr_type, nullable);
    // List arrays are unbound even with the fixed extents
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_EQ(array.extents().data(), (std::vector<std::uint64_t>{7}));
    EXPECT_EQ(array.volume(), 7);
    EXPECT_EQ(array.type(), arr_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);
    EXPECT_TRUE(array.nested());

    EXPECT_THROW(static_cast<void>(array.data()), std::invalid_argument);
    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_EQ(null_mask.extents(), array.extents());
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }

    auto list_array = array.as_list_array();
    // Sub-arrays of list arrays can be retrieved only when they are initialized first
    EXPECT_THROW(static_cast<void>(list_array.descriptor()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.vardata()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.child(2)), std::invalid_argument);
  }
  // Unbound
  {
    auto array = runtime->create_array(arr_type, 1, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
    EXPECT_EQ(array.type(), arr_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);
    EXPECT_TRUE(array.nested());

    EXPECT_THROW(static_cast<void>(array.data()), std::invalid_argument);
    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }

    auto list_array = array.as_list_array();
    // Sub-arrays of list arrays can be retrieved only when they are initialized first
    EXPECT_THROW(static_cast<void>(list_array.descriptor()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.vardata()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.child(2)), std::invalid_argument);
  }
  // create_list_array
  {
    auto type       = legate::list_type(legate::int64());
    auto descriptor = runtime->create_array(legate::Shape{7}, legate::rect_type(1), nullable);
    auto vardata    = runtime->create_array(legate::Shape{10}, legate::int64());
    auto array      = runtime->create_list_array(descriptor, vardata, type);

    EXPECT_FALSE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_EQ(array.extents().data(), (std::vector<std::uint64_t>{7}));
    EXPECT_EQ(array.volume(), 7);
    EXPECT_EQ(array.type(), type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);

    auto list_array = array.as_list_array();
    // Sub-arrays can be accessed
    static_cast<void>(list_array.descriptor());
    static_cast<void>(list_array.vardata());
  }
}

void test_string_array(bool nullable)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto str_type = legate::string_type();
  // Bound
  {
    auto array = runtime->create_array({5}, str_type, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_EQ(array.extents().data(), (std::vector<std::uint64_t>{5}));
    EXPECT_EQ(array.volume(), 5);
    EXPECT_EQ(array.type(), str_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);
    EXPECT_TRUE(array.nested());

    EXPECT_THROW(static_cast<void>(array.data()), std::invalid_argument);
    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_EQ(null_mask.extents(), array.extents());
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }

    auto list_array = array.as_string_array();
    // Sub-arrays of list arrays can be retrieved only when they are initialized first
    EXPECT_THROW(static_cast<void>(list_array.chars()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.offsets()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.child(2)), std::invalid_argument);
  }
  // Unbound
  {
    auto array = runtime->create_array(str_type, 1, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
    EXPECT_EQ(array.type(), str_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);
    EXPECT_TRUE(array.nested());

    EXPECT_THROW(static_cast<void>(array.data()), std::invalid_argument);
    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }

    auto list_array = array.as_string_array();
    // Sub-arrays of list arrays can be retrieved only when they are initialized first
    EXPECT_THROW(static_cast<void>(list_array.chars()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.offsets()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(list_array.child(2)), std::invalid_argument);
  }
}

void test_struct_array(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  auto st_type = legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
                   .as_struct_type();
  auto num_fields = st_type.num_fields();
  // Bound
  {
    auto array = runtime->create_array({4, 4}, st_type, nullable);
    EXPECT_FALSE(array.unbound());
    EXPECT_EQ(array.dim(), 2);
    EXPECT_EQ(array.extents().data(), (std::vector<std::uint64_t>{4, 4}));
    EXPECT_EQ(array.volume(), 16);
    EXPECT_EQ(array.type(), st_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), st_type.num_fields());
    EXPECT_TRUE(array.nested());

    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_EQ(null_mask.extents(), array.extents());
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }
    EXPECT_THROW(static_cast<void>(array.child(num_fields)), std::out_of_range);
    for (std::uint32_t idx = 0; idx < num_fields; ++idx) {
      auto field_type     = st_type.field_type(idx);
      auto field_subarray = array.child(idx);

      EXPECT_EQ(field_subarray.unbound(), array.unbound());
      EXPECT_EQ(field_subarray.dim(), array.dim());
      EXPECT_EQ(field_subarray.extents(), array.extents());
      EXPECT_EQ(field_subarray.volume(), array.volume());
      EXPECT_EQ(field_subarray.type(), field_type);
      // There'd be only one null mask for the whole struct array
      EXPECT_EQ(field_subarray.nullable(), false);
      EXPECT_THROW(static_cast<void>(field_subarray.null_mask()), std::invalid_argument);
      EXPECT_EQ(field_subarray.num_children(), 0);
    }
  }

  // Unbound
  {
    auto array = runtime->create_array(st_type, 3, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 3);
    EXPECT_EQ(array.type(), st_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), st_type.num_fields());

    if (!nullable) {
      EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
    } else {
      auto null_mask = array.null_mask();
      EXPECT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
      EXPECT_EQ(null_mask.type(), legate::bool_());
      EXPECT_EQ(null_mask.dim(), array.dim());
    }
    EXPECT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  }
}

void test_isomorphic(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // Bound
  {
    auto source  = runtime->create_array({4, 4}, legate::int64(), nullable);
    auto target1 = runtime->create_array_like(source);
    EXPECT_EQ(source.dim(), target1.dim());
    EXPECT_EQ(source.type(), target1.type());
    EXPECT_EQ(source.extents(), target1.extents());
    EXPECT_EQ(source.volume(), target1.volume());
    EXPECT_EQ(source.unbound(), target1.unbound());
    EXPECT_EQ(source.nullable(), target1.nullable());

    auto target2 = runtime->create_array_like(source, legate::float64());
    EXPECT_EQ(source.dim(), target2.dim());
    EXPECT_EQ(target2.type(), legate::float64());
    EXPECT_EQ(source.extents(), target2.extents());
    EXPECT_EQ(source.volume(), target2.volume());
    EXPECT_EQ(source.unbound(), target2.unbound());
    EXPECT_EQ(source.nullable(), target2.nullable());
  }
  // Unbound, Same type
  {
    auto source  = runtime->create_array(legate::int64(), 3, nullable);
    auto target1 = runtime->create_array_like(source);
    EXPECT_EQ(source.dim(), target1.dim());
    EXPECT_EQ(source.type(), target1.type());
    EXPECT_THROW(static_cast<void>(target1.extents()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(target1.volume()), std::invalid_argument);
    EXPECT_EQ(source.unbound(), target1.unbound());
    EXPECT_EQ(source.nullable(), target1.nullable());

    auto target2 = runtime->create_array_like(source, legate::float64());
    EXPECT_EQ(source.dim(), target2.dim());
    EXPECT_EQ(target2.type(), legate::float64());
    EXPECT_THROW(static_cast<void>(target2.extents()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(target2.volume()), std::invalid_argument);
    EXPECT_EQ(source.unbound(), target2.unbound());
    EXPECT_EQ(source.nullable(), target2.nullable());
  }
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional list/string arrays are not allowed
  EXPECT_THROW(static_cast<void>(runtime->create_array({1, 2, 3}, legate::string_type())),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(runtime->create_array({1, 2}, legate::list_type(legate::int64()))),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(runtime->create_array(legate::string_type(), 2)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(runtime->create_array(legate::list_type(legate::int64()), 3)),
               std::invalid_argument);

  // Invalid cast
  auto bound_array = runtime->create_array({4, 4}, legate::int64(), true);
  EXPECT_THROW(static_cast<void>(bound_array.as_list_array()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(bound_array.as_string_array()), std::invalid_argument);

  auto unbound_array = runtime->create_array(legate::int64(), 2, false);
  EXPECT_THROW(static_cast<void>(unbound_array.as_list_array()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(unbound_array.as_string_array()), std::invalid_argument);
}

void test_invalid_create_list_array()
{
  auto runtime = legate::Runtime::get_runtime();

  auto arr_unbound_rect1 = runtime->create_array(legate::rect_type(1));
  auto arr_unbound_int8  = runtime->create_array(legate::int8());
  auto arr_2d_rect1      = runtime->create_array(legate::Shape{10, 10}, legate::rect_type(1));
  auto arr_2d_int8       = runtime->create_array(legate::Shape{10, 10}, legate::int8());
  auto arr_nullable_int8 =
    runtime->create_array(legate::Shape{10}, legate::int8(), true /*nullable*/);
  auto arr_rect1 = runtime->create_array(legate::Shape{10}, legate::rect_type(1));
  auto arr_int8  = runtime->create_array(legate::Shape{10}, legate::int8());
  auto arr_int64 = runtime->create_array(legate::Shape{10}, legate::int64());

  // Tests for create_string_array
  {
    // Unbound sub-arrays
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_unbound_rect1, arr_int8)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_rect1, arr_unbound_int8)),
                 std::invalid_argument);
    // Multi-dimensional sub-arrays
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_2d_rect1, arr_int8)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_rect1, arr_2d_int8)),
                 std::invalid_argument);
    // Incorrect descriptor type
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_int64, arr_int8)),
                 std::invalid_argument);
    // Nullable vardata
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_rect1, arr_nullable_int8)),
                 std::invalid_argument);
    // Incorrect vardata type
    EXPECT_THROW(static_cast<void>(runtime->create_string_array(arr_rect1, arr_int64)),
                 std::invalid_argument);
  }

  // Tests for create_list_array
  {
    // Incorrect type
    EXPECT_THROW(
      static_cast<void>(runtime->create_list_array(arr_unbound_rect1, arr_int8, legate::int64())),
      std::invalid_argument);
    // Unbound sub-arrays
    EXPECT_THROW(static_cast<void>(runtime->create_list_array(arr_unbound_rect1, arr_int8)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(runtime->create_list_array(arr_rect1, arr_unbound_int8)),
                 std::invalid_argument);
    // Incorrect descriptor type
    EXPECT_THROW(static_cast<void>(runtime->create_list_array(arr_int64, arr_int8)),
                 std::invalid_argument);
    // Nullable vardata
    EXPECT_THROW(static_cast<void>(runtime->create_list_array(arr_rect1, arr_nullable_int8)),
                 std::invalid_argument);
    // Incorrect vardata type
    EXPECT_THROW(static_cast<void>(runtime->create_list_array(
                   arr_rect1, arr_int64, legate::list_type(legate::int8()))),
                 std::invalid_argument);
  }
}

void test_physical_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array({4, 4}, legate::int64(), nullable);
  auto array         = logical_array.get_physical_array();
  EXPECT_EQ(array.nullable(), nullable);
  EXPECT_EQ(array.dim(), 2);
  EXPECT_EQ(array.type(), legate::int64());
  EXPECT_FALSE(array.nested());
  EXPECT_EQ(array.shape<2>(), legate::Rect<2>({0, 0}, {3, 3}));
  EXPECT_EQ((array.domain().bounds<2, std::int64_t>()), legate::Rect<2>({0, 0}, {3, 3}));

  if (!nullable) {
    EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  } else {
    auto null_mask = array.null_mask();
    EXPECT_EQ(null_mask.shape<2>(), array.shape<2>());
    EXPECT_EQ(null_mask.domain(), array.domain());
    EXPECT_EQ(null_mask.type(), legate::bool_());
    EXPECT_EQ(null_mask.dim(), array.dim());
  }
  EXPECT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

void test_promote(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // primitive array
  {
    auto bound_array = runtime->create_array({1, 2, 3, 4}, legate::int64(), nullable);
    auto promoted    = bound_array.promote(0, 5);
    EXPECT_EQ(promoted.extents().data(), (std::vector<std::uint64_t>{5, 1, 2, 3, 4}));

    // Note: gitlab issue #16 unexpected behavior
    promoted = bound_array.promote(2, -1);
    EXPECT_EQ(promoted.extents().data(),
              (std::vector<std::uint64_t>{1, 2, static_cast<std::size_t>(-1), 3, 4}));

    // invalid extra_dim
    EXPECT_THROW(static_cast<void>(bound_array.promote(-1, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.promote(5, 3)), std::invalid_argument);

    auto unbound_array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.promote(1, 1)), std::invalid_argument);
  }

  // list array
  {
    auto list_type   = legate::list_type(legate::int64()).as_list_type();
    auto bound_array = runtime->create_array({7}, list_type, nullable);

    EXPECT_THROW(static_cast<void>(bound_array.promote(0, 10)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.promote(1, -1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.promote(-1, 3)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.promote(1, 3)), std::runtime_error);

    auto unbound_array = runtime->create_array(list_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.promote(1, 1)), std::runtime_error);
  }

  // string array
  {
    auto str_type    = legate::string_type();
    auto bound_array = runtime->create_array({6}, str_type, nullable);

    EXPECT_THROW(static_cast<void>(bound_array.promote(1, 10)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.promote(0, -1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.promote(-1, 3)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.promote(3, 3)), std::runtime_error);

    auto unbound_array = runtime->create_array(str_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.promote(1, 1)), std::runtime_error);
  }

  // struct array
  {
    auto st_type = legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
                     .as_struct_type();
    auto bound_array = runtime->create_array({4, 5, 6}, st_type, nullable);
    auto promoted    = bound_array.promote(2, 10);
    EXPECT_EQ(promoted.extents().data(), (std::vector<std::uint64_t>{4, 5, 10, 6}));

    // Note: gitlab issue #16 unexpected behavior
    promoted = bound_array.promote(1, -1);
    EXPECT_EQ(promoted.extents().data(),
              (std::vector<std::uint64_t>{4, static_cast<std::size_t>(-1), 5, 6}));

    // invalid extra_dim
    EXPECT_THROW(static_cast<void>(bound_array.promote(-1, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.promote(4, 3)), std::invalid_argument);

    auto unbound_array = runtime->create_array(st_type, 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.promote(1, 1)), std::invalid_argument);
  }
}

void test_project(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // primitive array
  {
    auto bound_array = runtime->create_array({1, 2, 3, 4}, legate::int64(), nullable);
    auto projected   = bound_array.project(0, 0);
    EXPECT_EQ(projected.extents().data(), (std::vector<std::uint64_t>{2, 3, 4}));

    // invalid dim
    EXPECT_THROW(static_cast<void>(bound_array.project(4, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.project(-3, 1)), std::invalid_argument);
    // invalid index
    EXPECT_THROW(static_cast<void>(bound_array.project(0, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.project(0, -4)), std::invalid_argument);

    auto unbound_array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.project(0, 0)), std::invalid_argument);
  }

  // list array
  {
    auto list_type   = legate::list_type(legate::int64()).as_list_type();
    auto bound_array = runtime->create_array({7}, list_type, nullable);

    EXPECT_THROW(static_cast<void>(bound_array.project(0, 5)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.project(1, -1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.project(-1, 3)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.project(1, 3)), std::runtime_error);

    auto unbound_array = runtime->create_array(list_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.project(0, 1)), std::runtime_error);
  }

  // string array
  {
    auto str_type    = legate::string_type();
    auto bound_array = runtime->create_array({6}, str_type, nullable);

    EXPECT_THROW(static_cast<void>(bound_array.project(0, 2)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.project(0, -1)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.project(-1, 3)), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.project(3, 3)), std::runtime_error);

    auto unbound_array = runtime->create_array(str_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.project(0, 1)), std::runtime_error);
  }

  // struct array
  {
    auto st_type = legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
                     .as_struct_type();
    auto bound_array = runtime->create_array({4, 5, 6}, st_type, nullable);
    auto projected   = bound_array.project(0, 2);
    EXPECT_EQ(projected.extents().data(), (std::vector<std::uint64_t>{5, 6}));

    // invalid dim
    EXPECT_THROW(static_cast<void>(bound_array.project(4, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.project(-3, 1)), std::invalid_argument);
    // invalid index
    EXPECT_THROW(static_cast<void>(bound_array.project(0, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.project(0, -4)), std::invalid_argument);

    auto unbound_array = runtime->create_array(st_type, 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.project(0, 2)), std::invalid_argument);
  }
}

void test_slice(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // primitive array
  {
    auto bound_array = runtime->create_array({1, 2, 3, 4}, legate::int64(), nullable);
    // slice [OPEN, STOP) of dim i
    auto sliced = bound_array.slice(2, legate::Slice(-2, -1));
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{1, 2, 1, 4}));

    sliced = bound_array.slice(2, legate::Slice(1, 2));
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{1, 2, 1, 4}));

    // full slice
    sliced = bound_array.slice(0, legate::Slice());
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{1, 2, 3, 4}));

    // out of bounds
    EXPECT_THROW(static_cast<void>(bound_array.slice(2, legate::Slice(1, 4))),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.slice(2, legate::Slice(3, 4))),
                 std::invalid_argument);

    // invalid dim
    EXPECT_THROW(static_cast<void>(bound_array.slice(4, legate::Slice(1, 3))),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.slice(-2, legate::Slice(1, 3))),
                 std::invalid_argument);

    // Note: gitlab issue #16 unexpected: sliced.extents() Which is: (1,18446744073709551615,3,4,)
    // sliced = bound_array.slice(1, legate::Slice(-1, 0));
    // sliced = bound_array.slice(1, legate::Slice(10, 8));

    // Note: gitlab issue #16 crashes
    // sliced = bound_array.slice(1, legate::Slice(-9, -8));
    // sliced = bound_array.slice(1, legate::Slice(-8, -10));
    // sliced = bound_array.slice(1, legate::Slice(0, 0));
    // sliced = bound_array.slice(1, legate::Slice(-1, 1));

    auto unbound_array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.slice(1, legate::Slice(0, 1))),
                 std::invalid_argument);
  }

  // list array
  {
    auto list_type   = legate::list_type(legate::int64()).as_list_type();
    auto bound_array = runtime->create_array({7}, list_type, nullable);

    EXPECT_THROW(static_cast<void>(bound_array.slice(0, legate::Slice(0, 1))), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.slice(1, legate::Slice(1, 3))), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.slice(-1, legate::Slice(-1, 3))),
                 std::runtime_error);

    auto unbound_array = runtime->create_array(list_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.slice(0, legate::Slice())), std::runtime_error);
  }

  // string array
  {
    auto str_type    = legate::string_type();
    auto bound_array = runtime->create_array({6}, str_type, nullable);

    EXPECT_THROW(static_cast<void>(bound_array.slice(0, legate::Slice(0, 1))), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.slice(1, legate::Slice(0, 1))), std::runtime_error);
    EXPECT_THROW(static_cast<void>(bound_array.slice(-1, legate::Slice(0, 1))), std::runtime_error);

    auto unbound_array = runtime->create_array(str_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.slice(0, legate::Slice(0, 0))),
                 std::runtime_error);
  }

  // struct array
  {
    auto st_type = legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
                     .as_struct_type();
    auto bound_array = runtime->create_array({4, 5, 6}, st_type, nullable);
    // slice [OPEN, STOP) of dim i
    auto sliced = bound_array.slice(2, legate::Slice(-2, -1));
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{4, 5, 1}));

    sliced = bound_array.slice(2, legate::Slice(1, 2));
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{4, 5, 1}));

    sliced = bound_array.slice(2, legate::Slice(1, 4));
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{4, 5, 3}));

    sliced = bound_array.slice(2, legate::Slice(3, 4));
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{4, 5, 1}));

    // full slice
    sliced = bound_array.slice(0, legate::Slice());
    EXPECT_EQ(sliced.extents().data(), (std::vector<std::uint64_t>{4, 5, 6}));

    // invalid dim
    EXPECT_THROW(static_cast<void>(bound_array.slice(4, legate::Slice(1, 3))),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.slice(-2, legate::Slice(1, 3))),
                 std::invalid_argument);

    // Note: gitlab issue #16 crashes
    // sliced = bound_array.slice(1, legate::Slice(-9, -8));
    // sliced = bound_array.slice(1, legate::Slice(-8, -10));
    // sliced = bound_array.slice(1, legate::Slice(0, 0));
    // sliced = bound_array.slice(1, legate::Slice(-1, 1));

    // Note: gitlab issue #16 unexpected: sliced.extents() Which is: (1,18446744073709551615,3,4,)
    // sliced = bound_array.slice(1, legate::Slice(-1, 0));
    // sliced = bound_array.slice(1, legate::Slice(10, 8));

    auto unbound_array = runtime->create_array(st_type, 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.slice(1, legate::Slice(0, 1))),
                 std::invalid_argument);
  }
}

void test_transpose(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // primitive array
  {
    auto bound_array = runtime->create_array({1, 2, 3, 4}, legate::int64(), nullable);
    auto transposed  = bound_array.transpose({1, 0, 3, 2});
    EXPECT_EQ(transposed.extents().data(), (std::vector<std::uint64_t>{2, 1, 4, 3}));

    // invalid axes length
    EXPECT_THROW(static_cast<void>(bound_array.transpose({2})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({2, 1, 0, 3, 4})), std::invalid_argument);
    // axes has duplicates
    EXPECT_THROW(static_cast<void>(bound_array.transpose({0, 0, 2, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({-1, -1, 1, 2})), std::invalid_argument);
    // invalid axis in axes
    EXPECT_THROW(static_cast<void>(bound_array.transpose({4, 0, 1, 2})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({-1, 0, 1, 2})), std::invalid_argument);

    auto unbound_array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.transpose({1, 0, 3, 2})), std::invalid_argument);
  }

  // list array
  {
    auto list_type   = legate::list_type(legate::int64()).as_list_type();
    auto bound_array = runtime->create_array({7}, list_type, nullable);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({0})), std::runtime_error);

    auto unbound_array = runtime->create_array(list_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.transpose({0})), std::runtime_error);
  }

  // string array
  {
    auto str_type    = legate::string_type();
    auto bound_array = runtime->create_array({6}, str_type, nullable);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({0})), std::runtime_error);

    auto unbound_array = runtime->create_array(str_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.transpose({0})), std::runtime_error);
  }

  // struct array
  {
    auto st_type = legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
                     .as_struct_type();
    auto bound_array = runtime->create_array({4, 5, 6}, st_type, nullable);
    auto transposed  = bound_array.transpose({1, 0, 2});
    EXPECT_EQ(transposed.extents().data(), (std::vector<std::uint64_t>{5, 4, 6}));

    // invalid axes length
    EXPECT_THROW(static_cast<void>(bound_array.transpose({0, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({2, 1, 0, 3})), std::invalid_argument);
    // axes has duplicates
    EXPECT_THROW(static_cast<void>(bound_array.transpose({0, 0, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({-1, -1, 0})), std::invalid_argument);
    // invalid axis in axes
    EXPECT_THROW(static_cast<void>(bound_array.transpose({0, 1, 3})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.transpose({-1, 0, 1})), std::invalid_argument);

    auto unbound_array = runtime->create_array(st_type, 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.transpose({1, 0, 2})), std::invalid_argument);
  }
}

void test_delinearize(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // primitive array
  {
    auto bound_array  = runtime->create_array({1, 2, 3, 4}, legate::int64(), nullable);
    auto delinearized = bound_array.delinearize(0, {1, 1});
    EXPECT_EQ(delinearized.extents().data(), (std::vector<std::uint64_t>{1, 1, 2, 3, 4}));

    delinearized = bound_array.delinearize(3, {4});
    EXPECT_EQ(delinearized.extents().data(), (std::vector<std::uint64_t>{1, 2, 3, 4}));

    delinearized = bound_array.delinearize(3, {2, 1, 2, 1});
    EXPECT_EQ(delinearized.extents().data(), (std::vector<std::uint64_t>{1, 2, 3, 2, 1, 2, 1}));

    // invalid dim
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(4, {1, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(-1, {1, 1})), std::invalid_argument);
    // invalid sizes
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {1, 2})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {-1UL, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(3, {2})), std::invalid_argument);

    // Note: gitlab issue #16 unexpected: delinearized.extents is
    // (18446744073709551615,18446744073709551615,2,3,4,)
    // delinearized = bound_array.delinearize(0, (std::vector<std::int64_t>{-1, -1}));

    auto unbound_array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.delinearize(0, {1, 1})), std::invalid_argument);
  }

  // list array
  {
    auto list_type   = legate::list_type(legate::int64()).as_list_type();
    auto bound_array = runtime->create_array({7}, list_type, nullable);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {7})), std::runtime_error);

    auto unbound_array = runtime->create_array(list_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.delinearize(0, {7})), std::runtime_error);
  }

  // string array
  {
    auto str_type    = legate::string_type();
    auto bound_array = runtime->create_array({6}, str_type, nullable);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {6})), std::runtime_error);

    auto unbound_array = runtime->create_array(str_type, 1, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.delinearize(0, {6})), std::runtime_error);
  }

  // struct array
  {
    auto st_type = legate::struct_type(true, legate::uint16(), legate::int64(), legate::float32())
                     .as_struct_type();
    auto bound_array  = runtime->create_array({4, 5, 6}, st_type, nullable);
    auto delinearized = bound_array.delinearize(0, {2, 2});
    EXPECT_EQ(delinearized.extents().data(), (std::vector<std::uint64_t>{2, 2, 5, 6}));

    delinearized = bound_array.delinearize(2, {6});
    EXPECT_EQ(delinearized.extents().data(), (std::vector<std::uint64_t>{4, 5, 6}));

    delinearized = bound_array.delinearize(2, {2, 3, 1});
    EXPECT_EQ(delinearized.extents().data(), (std::vector<std::uint64_t>{4, 5, 2, 3, 1}));

    // invalid dim
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(4, {1, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(-1, {1, 1})), std::invalid_argument);
    // invalid sizes
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {1, 2})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {-1UL, 1})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bound_array.delinearize(2, {2})), std::invalid_argument);

    EXPECT_THROW(static_cast<void>(bound_array.delinearize(0, {-2UL, -2UL})),
                 std::invalid_argument);

    auto unbound_array = runtime->create_array(st_type, 3, nullable);
    EXPECT_THROW(static_cast<void>(unbound_array.delinearize(0, {1, 1})), std::invalid_argument);
  }
}

TEST_F(LogicalArrayUnit, CreatePrimitiveNonNullable) { test_primitive_array(false); }

TEST_F(LogicalArrayUnit, CreatePrimitiveNullable) { test_primitive_array(true); }

TEST_F(LogicalArrayUnit, CreateListNonNullable) { test_list_array(false); }

TEST_F(LogicalArrayUnit, CreateListNullable) { test_list_array(true); }

TEST_F(LogicalArrayUnit, CreateStringNonNullable) { test_string_array(false); }

TEST_F(LogicalArrayUnit, CreateStringNullable) { test_string_array(true); }

TEST_F(LogicalArrayUnit, CreateStructNonNullable) { test_struct_array(false); }

TEST_F(LogicalArrayUnit, CreateStructNullable) { test_struct_array(true); }

TEST_F(LogicalArrayUnit, CreateLike)
{
  test_isomorphic(false);
  test_isomorphic(true);
}

TEST_F(LogicalArrayUnit, CreateInvalid) { test_invalid(); }

TEST_F(LogicalArrayUnit, CreateInvalidList) { test_invalid_create_list_array(); }

TEST_F(LogicalArrayUnit, PhsicalArray)
{
  test_physical_array(true);
  test_physical_array(false);
}

TEST_F(LogicalArrayUnit, Promote)
{
  test_promote(true);
  test_promote(false);
}

TEST_F(LogicalArrayUnit, Project)
{
  test_project(true);
  test_project(false);
}

TEST_F(LogicalArrayUnit, Slice)
{
  test_slice(true);
  test_slice(false);
}

TEST_F(LogicalArrayUnit, Transpose)
{
  test_transpose(true);
  test_transpose(false);
}

TEST_F(LogicalArrayUnit, Delinearize)
{
  test_delinearize(true);
  test_delinearize(false);
}

}  // namespace logical_array_test
