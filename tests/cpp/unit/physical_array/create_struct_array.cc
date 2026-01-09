/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_create_struct_test {

namespace {

using CreateStructPhysicalArrayUnit = DefaultFixture;

class NullableCreateStructArrayTest : public CreateStructPhysicalArrayUnit,
                                      public ::testing::WithParamInterface<bool> {};

class BoundPhysicalStructArrayTest
  : public CreateStructPhysicalArrayUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape, legate::Type, bool, legate::Rect<3>>> {};

INSTANTIATE_TEST_SUITE_P(CreateStructPhysicalArrayUnit,
                         NullableCreateStructArrayTest,
                         ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
  CreateStructPhysicalArrayUnit,
  BoundPhysicalStructArrayTest,
  ::testing::Values(std::make_tuple(legate::Shape{1, 2, 4},
                                    legate::struct_type(true, legate::int64(), legate::float32()),
                                    true,
                                    legate::Rect<3>{{0, 0, 0}, {0, 1, 3}}),
                    std::make_tuple(legate::Shape{4, 5, 6},
                                    legate::struct_type(true, legate::int64(), legate::float32()),
                                    true,
                                    legate::Rect<3>{{0, 0, 0}, {3, 4, 5}}),
                    std::make_tuple(legate::Shape{4, 5, 6},
                                    legate::struct_type(false, legate::int64(), legate::float32()),
                                    true,
                                    legate::Rect<3>{{0, 0, 0}, {3, 4, 5}}),
                    std::make_tuple(legate::Shape{1, 2, 4},
                                    legate::struct_type(false, legate::int64(), legate::float32()),
                                    true,
                                    legate::Rect<3>{{0, 0, 0}, {0, 1, 3}})));

}  // namespace

TEST_P(BoundPhysicalStructArrayTest, Create)
{
  const auto [shape, type, nullable, bound_rect] = GetParam();
  auto runtime                                   = legate::Runtime::get_runtime();
  auto logical_array                             = runtime->create_array(shape, type, nullable);
  auto array                                     = logical_array.get_physical_array();
  static constexpr std::int32_t DIM              = 3;

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type().code(), legate::Type::Code::STRUCT);
  ASSERT_TRUE(array.nested());
  ASSERT_EQ(array.shape<DIM>(), bound_rect);
  ASSERT_EQ((array.domain().bounds<DIM, std::int64_t>()), bound_rect);

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.shape<DIM>(), array.shape<DIM>());
    ASSERT_EQ(null_mask.domain(), array.domain());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto field_subarray1 = array.child(0);

  ASSERT_FALSE(field_subarray1.nullable());
  ASSERT_EQ(field_subarray1.dim(), DIM);
  ASSERT_EQ(field_subarray1.type(), legate::int64());
  ASSERT_EQ((field_subarray1.domain().bounds<DIM, std::int64_t>()), bound_rect);

  auto field_subarray2 = array.child(1);

  ASSERT_FALSE(field_subarray2.nullable());
  ASSERT_EQ(field_subarray2.dim(), DIM);
  ASSERT_EQ(field_subarray2.type(), legate::float32());
  ASSERT_EQ((field_subarray2.domain().bounds<DIM, std::int64_t>()), bound_rect);
}

TEST_P(NullableCreateStructArrayTest, InvalidBoundStructArrayChild)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array(
    legate::Shape{1}, legate::struct_type(true, legate::int64(), legate::float32()), GetParam());
  auto array = logical_array.get_physical_array();

  ASSERT_THROW(static_cast<void>(array.child(2)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(array.child(-1)), std::out_of_range);
}

TEST_P(NullableCreateStructArrayTest, InvalidCastBoundStructArray)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array(
    legate::Shape{1}, legate::struct_type(true, legate::int64(), legate::float32()), GetParam());
  auto array = logical_array.get_physical_array();

  ASSERT_THROW(static_cast<void>(array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

}  // namespace physical_array_create_struct_test
