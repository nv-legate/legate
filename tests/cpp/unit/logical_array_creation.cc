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
#include "utilities/utilities.h"

namespace array_test {

using LogicalArray = DefaultFixture;

void test_primitive_array(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // Bound
  {
    auto array = runtime->create_array(legate::Shape{4, 4}, legate::int64(), nullable);
    EXPECT_FALSE(array.unbound());
    EXPECT_EQ(array.dim(), 2);
    EXPECT_EQ(array.extents().data(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(array.volume(), 16);
    EXPECT_EQ(array.type(), legate::int64());
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 0);

    auto store = array.data();
    EXPECT_FALSE(store.unbound());
    EXPECT_EQ(store.dim(), 2);
    EXPECT_EQ(store.extents().data(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(store.volume(), 16);
    EXPECT_EQ(store.type(), legate::int64());

    if (!nullable) { EXPECT_THROW(array.null_mask(), std::invalid_argument); }
    EXPECT_THROW(array.child(0), std::invalid_argument);
  }
  // Unbound
  {
    auto array = runtime->create_array(legate::int64(), 3, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 3);
    EXPECT_THROW(array.extents(), std::invalid_argument);
    EXPECT_THROW(array.volume(), std::invalid_argument);
    EXPECT_EQ(array.type(), legate::int64());
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 0);

    auto store = array.data();
    EXPECT_TRUE(store.unbound());
    EXPECT_EQ(store.dim(), 3);
    EXPECT_THROW(store.extents(), std::invalid_argument);
    EXPECT_THROW(store.volume(), std::invalid_argument);
    EXPECT_EQ(store.type(), legate::int64());

    if (!nullable) { EXPECT_THROW(array.null_mask(), std::invalid_argument); }
    EXPECT_THROW(array.child(0), std::invalid_argument);
  }
}

void test_list_array(bool nullable)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto arr_type = legate::list_type(legate::int64()).as_list_type();
  // Bound descriptor
  {
    auto array = runtime->create_array(legate::Shape{7}, arr_type, nullable);
    // List arrays are unbound even with the fixed extents
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_EQ(array.extents().data(), (std::vector<size_t>{7}));
    EXPECT_EQ(array.volume(), 7);
    EXPECT_EQ(array.type(), arr_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);

    EXPECT_THROW(array.data(), std::invalid_argument);
    if (!nullable) { EXPECT_THROW(array.null_mask(), std::invalid_argument); }

    auto list_array = array.as_list_array();
    // Sub-arrays of list arrays can be retrieved only when they are initialized first
    EXPECT_THROW(list_array.descriptor(), std::invalid_argument);
    EXPECT_THROW(list_array.vardata(), std::invalid_argument);
    EXPECT_THROW(list_array.child(2), std::invalid_argument);
  }
  // Unbound
  {
    auto array = runtime->create_array(arr_type, 1, nullable);
    EXPECT_TRUE(array.unbound());
    EXPECT_EQ(array.dim(), 1);
    EXPECT_EQ(array.type(), arr_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), 2);

    EXPECT_THROW(array.data(), std::invalid_argument);
    if (!nullable) { EXPECT_THROW(array.null_mask(), std::invalid_argument); }

    auto list_array = array.as_list_array();
    // Sub-arrays of list arrays can be retrieved only when they are initialized first
    EXPECT_THROW(list_array.descriptor(), std::invalid_argument);
    EXPECT_THROW(list_array.vardata(), std::invalid_argument);
    EXPECT_THROW(list_array.child(2), std::invalid_argument);
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
    auto array = runtime->create_array(legate::Shape{4, 4}, st_type, nullable);
    EXPECT_FALSE(array.unbound());
    EXPECT_EQ(array.dim(), 2);
    EXPECT_EQ(array.extents().data(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(array.volume(), 16);
    EXPECT_EQ(array.type(), st_type);
    EXPECT_EQ(array.nullable(), nullable);
    EXPECT_EQ(array.num_children(), st_type.num_fields());

    if (!nullable) { EXPECT_THROW(array.null_mask(), std::invalid_argument); }
    EXPECT_THROW(array.child(num_fields), std::out_of_range);
    for (uint32_t idx = 0; idx < num_fields; ++idx) {
      auto field_type     = st_type.field_type(idx);
      auto field_subarray = array.child(idx);

      EXPECT_EQ(field_subarray.unbound(), array.unbound());
      EXPECT_EQ(field_subarray.dim(), array.dim());
      EXPECT_EQ(field_subarray.extents(), array.extents());
      EXPECT_EQ(field_subarray.volume(), array.volume());
      EXPECT_EQ(field_subarray.type(), field_type);
      // There'd be only one null mask for the whole struct array
      EXPECT_EQ(field_subarray.nullable(), false);
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

    if (!nullable) { EXPECT_THROW(array.null_mask(), std::invalid_argument); }
    EXPECT_THROW(array.child(0), std::invalid_argument);
  }
}

void test_isomorphic(bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();
  // Bound
  {
    auto source  = runtime->create_array(legate::Shape{4, 4}, legate::int64(), nullable);
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
    EXPECT_THROW(target1.extents(), std::invalid_argument);
    EXPECT_THROW(target1.volume(), std::invalid_argument);
    EXPECT_EQ(source.unbound(), target1.unbound());
    EXPECT_EQ(source.nullable(), target1.nullable());

    auto target2 = runtime->create_array_like(source, legate::float64());
    EXPECT_EQ(source.dim(), target2.dim());
    EXPECT_EQ(target2.type(), target2.type());
    EXPECT_THROW(target2.extents(), std::invalid_argument);
    EXPECT_THROW(target2.volume(), std::invalid_argument);
    EXPECT_EQ(source.unbound(), target2.unbound());
    EXPECT_EQ(source.nullable(), target2.nullable());
  }
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional list/string arrays are not allowed
  EXPECT_THROW(
    static_cast<void>(runtime->create_array(legate::Shape{1, 2, 3}, legate::string_type())),
    std::invalid_argument);
  EXPECT_THROW(static_cast<void>(
                 runtime->create_array(legate::Shape{1, 2}, legate::list_type(legate::int64()))),
               std::invalid_argument);
  EXPECT_THROW((void)runtime->create_array(legate::string_type(), 2), std::invalid_argument);
  EXPECT_THROW((void)runtime->create_array(legate::list_type(legate::int64()), 3),
               std::invalid_argument);
}

TEST_F(LogicalArray, CreatePrimitiveNonNullable) { test_primitive_array(false); }

TEST_F(LogicalArray, CreatePrimitiveNullable) { test_primitive_array(true); }

TEST_F(LogicalArray, CreateListNonNullable) { test_list_array(false); }

TEST_F(LogicalArray, CreateListNullable) { test_list_array(true); }

TEST_F(LogicalArray, CreateStructNonNullable) { test_struct_array(false); }

TEST_F(LogicalArray, CreateStructNullable) { test_struct_array(true); }

TEST_F(LogicalArray, CreateLike)
{
  test_isomorphic(false);
  test_isomorphic(true);
}

TEST_F(LogicalArray, CreateInvalid) { test_invalid(); }

}  // namespace array_test
