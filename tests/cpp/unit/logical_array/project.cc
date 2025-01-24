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

namespace logical_array_project_test {

namespace {

using LogicalArrayProjectUnit = DefaultFixture;

// project test for bound fixed size array: primitive type array and struct type array
class ProjectFixedArrayTest
  : public LogicalArrayProjectUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type,
                 bool,
                 std::tuple<std::int32_t, std::int64_t, std::vector<std::uint64_t>>>> {};

// project test for arrays not able to perform delinearize: unbound array, variable size array: list
// array and string array
class NonProjectTest : public LogicalArrayProjectUnit,
                       public ::testing::WithParamInterface<std::tuple<legate::Type, bool>> {};

// negative cases for projecting bound fixed size array and bound variable array
class NegativeProjectTest
  : public LogicalArrayProjectUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type, bool, std::tuple<std::int32_t, std::int64_t>>> {};

class NegativeProjectDimTest : public NegativeProjectTest {};

class NegativeProjectIndexTest : public NegativeProjectTest {};

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayProjectUnit,
  ProjectFixedArrayTest,
  ::testing::Combine(
    ::testing::Values(legate::int64(), logical_array_util_test::struct_type()),
    ::testing::Values(true, false),
    ::testing::Values(std::make_tuple(0, 0, std::vector<std::uint64_t>({2, 3, 4})),
                      std::make_tuple(1, 1, std::vector<std::uint64_t>({1, 3, 4})),
                      std::make_tuple(2, 1, std::vector<std::uint64_t>({1, 2, 4})),
                      std::make_tuple(3, 3, std::vector<std::uint64_t>({1, 2, 3})))));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayProjectUnit,
  NonProjectTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayProjectUnit,
  NegativeProjectDimTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(4, 1), std::make_tuple(-3, 1))));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayProjectUnit,
  NegativeProjectIndexTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(0, 100), std::make_tuple(0, -4))));

void test_negative_project(
  const legate::Type& type, bool bound, bool nullable, std::int32_t dim, std::int64_t index)
{
  auto array = logical_array_util_test::create_array_with_type(type, bound, nullable);

  if (type.variable_size()) {
    ASSERT_THROW(static_cast<void>(array.project(dim, index)), std::runtime_error);
  } else {
    ASSERT_THROW(static_cast<void>(array.project(dim, index)), std::invalid_argument);
  }
}

}  // namespace

TEST_P(ProjectFixedArrayTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, index, shape]      = params;
  auto bound_array = logical_array_util_test::create_array_with_type(type, true, nullable);
  auto projected   = bound_array.project(dim, index);

  ASSERT_EQ(projected.extents().data(), shape);
}

TEST_P(NonProjectTest, Basic)
{
  const auto [type, nullable] = GetParam();

  test_negative_project(type, false, nullable, 0, 0);
}

TEST_P(NegativeProjectDimTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, index]             = params;

  test_negative_project(type, true, nullable, dim, index);
}

TEST_P(NegativeProjectIndexTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, index]             = params;

  test_negative_project(type, true, nullable, dim, index);
}

}  // namespace logical_array_project_test
