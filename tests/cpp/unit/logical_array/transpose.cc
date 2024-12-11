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

#include "legate.h"
#include "utilities/utilities.h"
#include "utils.h"

#include <gtest/gtest.h>

namespace logical_array_transpose_test {

namespace {

using LogicalArrayTransposeUnit = DefaultFixture;

// transpose test for bound fixed size array: primitive type array and struct type array
class TransposeFixedArrayTest
  : public LogicalArrayTransposeUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type,
                 bool,
                 std::tuple<std::vector<std::int32_t>, std::vector<std::uint64_t>>>> {};

// transpose test for arrays not able to perform delinearize: unbound array, variable size array:
// list array and string array
class NonTransposeTest : public LogicalArrayTransposeUnit,
                         public ::testing::WithParamInterface<std::tuple<legate::Type, bool>> {};

// negative cases for transposing bound fixed size array and bound variable array
class NegativeTransposeAxesTest : public LogicalArrayTransposeUnit,
                                  public ::testing::WithParamInterface<
                                    std::tuple<legate::Type, bool, std::vector<std::int32_t>>> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayTransposeUnit,
  TransposeFixedArrayTest,
  ::testing::Combine(::testing::Values(legate::int64(), logical_array_util_test::struct_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(std::vector<std::int32_t>({1, 0, 3, 2}),
                                                       std::vector<std::uint64_t>({2, 1, 4, 3})),
                                       std::make_tuple(std::vector<std::int32_t>({1, 3, 0, 2}),
                                                       std::vector<std::uint64_t>({2, 4, 1, 3})),
                                       std::make_tuple(std::vector<std::int32_t>({1, 3, 2, 0}),
                                                       std::vector<std::uint64_t>({2, 4, 3, 1})))));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayTransposeUnit,
  NonTransposeTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayTransposeUnit,
  NegativeTransposeAxesTest,
  ::testing::Combine(
    ::testing::Values(legate::uint16(),
                      logical_array_util_test::struct_type(),
                      logical_array_util_test::list_type(),
                      legate::string_type()),
    ::testing::Values(true, false),
    ::testing::Values(std::vector<std::int32_t>({2, 1, 0, 3, 4}) /* invalid axes length */,
                      std::vector<std::int32_t>({0, 0, 2, 1}) /* axes has duplicates */,
                      std::vector<std::int32_t>({4, 0, 1, 2}) /* invalid axis in axes */,
                      std::vector<std::int32_t>({3, -1, 1, 2}) /* invalid axis in axes */)));

void test_negative_transpose(const legate::Type& type,
                             bool bound,
                             bool nullable,
                             const std::vector<std::int32_t>& axes)
{
  auto array = logical_array_util_test::create_array_with_type(type, bound, nullable);

  if (type.variable_size()) {
    ASSERT_THROW(static_cast<void>(array.transpose(axes)), std::runtime_error);
  } else {
    ASSERT_THROW(static_cast<void>(array.transpose(axes)), std::invalid_argument);
  }
}

}  // namespace

TEST_P(TransposeFixedArrayTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [axes, shape]            = params;
  auto bound_array = logical_array_util_test::create_array_with_type(type, true, nullable);
  auto transposed  = bound_array.transpose(axes);

  ASSERT_EQ(transposed.extents().data(), shape);
}

TEST_P(NonTransposeTest, Basic)
{
  const auto [type, nullable] = GetParam();

  test_negative_transpose(type, false, nullable, {0});
}

TEST_P(NegativeTransposeAxesTest, Basic)
{
  const auto [type, nullable, axes] = GetParam();

  test_negative_transpose(type, true, nullable, axes);
}

}  // namespace logical_array_transpose_test
