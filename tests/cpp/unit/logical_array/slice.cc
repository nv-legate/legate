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

namespace logical_array_slice_test {

namespace {

using LogicalArraySliceUnit = DefaultFixture;

// slice test for bound fixed size array: primitive type array and struct type array
class SliceFixedArrayTest
  : public LogicalArraySliceUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type,
                 bool,
                 std::tuple<std::int32_t, legate::Slice, std::vector<std::uint64_t>>>> {};

// slice test for arrays not able to perform delinearize: unbound array, variable size array: list
// array and string array
class NonSliceTest : public LogicalArraySliceUnit,
                     public ::testing::WithParamInterface<std::tuple<legate::Type, bool>> {};

// negative cases for slicing bound fixed size array and bound variable array
class NegativeSliceTest
  : public LogicalArraySliceUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type, bool, std::tuple<std::int32_t, legate::Slice>>> {};

class NegativeSliceDimTest : public NegativeSliceTest {};

class NegativeSlieBoundsTest : public NegativeSliceTest {};

INSTANTIATE_TEST_SUITE_P(
  LogicalArraySliceUnit,
  SliceFixedArrayTest,
  ::testing::Combine(
    ::testing::Values(legate::int64(), logical_array_util_test::struct_type()),
    ::testing::Values(true, false),
    ::testing::Values(
      std::make_tuple(1,
                      legate::Slice(0, 0),
                      std::vector<std::uint64_t>(
                        {1, 0, 3, 4})) /*  start = stop = 0, slice [OPEN, STOP) of dim i */,
      std::make_tuple(
        2, legate::Slice(), std::vector<std::uint64_t>({1, 2, 3, 4})) /* full slice */,
      std::make_tuple(
        2,
        legate::Slice(-2, -1),
        std::vector<std::uint64_t>(
          {1, 2, 1, 4})) /* start < stop < 0, 0 < start + extent[dim] < stop + extent[dim] */,
      std::make_tuple(2,
                      legate::Slice(-9, -2),
                      std::vector<std::uint64_t>(
                        {1, 2, 1, 4})) /* start < stop < 0, start + extent[dim] < 0 < stop +
                                          extent[dim], stop - start > extent[dim] */
      ,
      std::make_tuple(
        0,
        legate::Slice(-1, 0),
        std::vector<std::uint64_t>({0, 2, 3, 4})) /* start < 0 = stop, start + extent[dim] = 0 */,
      std::make_tuple(
        0,
        legate::Slice(0, -1),
        std::vector<std::uint64_t>({0, 2, 3, 4})) /* start = 0 > stop, stop + extent[dim] = 0 */,
      std::make_tuple(1,
                      legate::Slice(-2, 1),
                      std::vector<std::uint64_t>(
                        {1, 1, 3, 4})) /* start < 0 < stop, start + extent[dim] < stop */,
      std::make_tuple(3,
                      legate::Slice(-2, 2),
                      std::vector<std::uint64_t>(
                        {1, 2, 3, 0})) /* start < 0 < stop, start + extent[dim] = stop */,
      std::make_tuple(2,
                      legate::Slice(-1, 1),
                      std::vector<std::uint64_t>(
                        {1, 2, 0, 4})) /* start < 0 < stop, start + extent[dim] > stop */,
      std::make_tuple(
        2, legate::Slice(1, 3), std::vector<std::uint64_t>({1, 2, 2, 4})) /* 0 < start < stop */,
      std::make_tuple(
        1, legate::Slice(10, 8), std::vector<std::uint64_t>({1, 0, 3, 4})) /* start > stop > 0 */,
      std::make_tuple(
        2,
        legate::Slice(1, -1),
        std::vector<std::uint64_t>({1, 2, 1, 4})) /* start > 0 > stop, start < top + extent[dim] */,
      std::make_tuple(
        2,
        legate::Slice(1, -2),
        std::vector<std::uint64_t>({1, 2, 0, 4})) /* start > 0 > stop, start = top + extent[dim] */,
      std::make_tuple(
        2,
        legate::Slice(2, -3),
        std::vector<std::uint64_t>({1, 2, 0, 4})) /* start > 0 > stop, start > top + extent[dim] */,
      std::make_tuple(2,
                      legate::Slice(-8, -10),
                      std::vector<std::uint64_t>({1, 2, 0, 4})) /* 0 > start > stop */)));

// TODO(joyshennv): issue #1481
// std::make_tuple(2, legate::Slice(-9, -8), std::vector<std::uint64_t>({1, 2, 1, 4})) /* start <
// stop < 0, start + extent[dim] < stop + extent[dim] < 0, fail, got {1, 2, 0, 4} */

INSTANTIATE_TEST_SUITE_P(
  LogicalArraySliceUnit,
  NonSliceTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(
  LogicalArraySliceUnit,
  NegativeSlieBoundsTest,
  ::testing::Combine(
    ::testing::Values(legate::uint16(),
                      logical_array_util_test::struct_type(),
                      logical_array_util_test::list_type(),
                      legate::string_type()),
    ::testing::Values(true, false),
    ::testing::Values(std::make_tuple(2, legate::Slice(1, 4)) /* stop > extent[dim] */,
                      std::make_tuple(2, legate::Slice(-3, 4)) /* stop > extent[dim] */,
                      std::make_tuple(1, legate::Slice(3, 4)) /* satrt = extent[dim] */,
                      std::make_tuple(1, legate::Slice(4, 5)) /* satrt > extent[dim] */)));

INSTANTIATE_TEST_SUITE_P(
  LogicalArraySliceUnit,
  NegativeSliceDimTest,
  ::testing::Combine(::testing::Values(legate::uint16(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(4, legate::Slice(1, 3)),
                                       std::make_tuple(-2, legate::Slice(1, 2)))));

void test_negative_slice(
  const legate::Type& type, bool bound, bool nullable, std::int32_t dim, const legate::Slice& slice)
{
  auto array = logical_array_util_test::create_array_with_type(type, bound, nullable);

  if (type.variable_size()) {
    ASSERT_THROW(static_cast<void>(array.slice(dim, slice)), std::runtime_error);
  } else {
    ASSERT_THROW(static_cast<void>(array.slice(dim, slice)), std::invalid_argument);
  }
}

}  // namespace

TEST_P(SliceFixedArrayTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, slice, shape]      = params;
  auto bound_array = logical_array_util_test::create_array_with_type(type, true, nullable);
  auto sliced      = bound_array.slice(dim, slice);

  ASSERT_EQ(sliced.extents().data(), shape);
}

TEST_P(NonSliceTest, Basic)
{
  const auto [type, nullable] = GetParam();

  test_negative_slice(type, false, nullable, 0, legate::Slice());
}

TEST_P(NegativeSliceDimTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, slice]             = params;

  test_negative_slice(type, true, nullable, dim, slice);
}

TEST_P(NegativeSlieBoundsTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, slice]             = params;

  test_negative_slice(type, true, nullable, dim, slice);
}

}  // namespace logical_array_slice_test
