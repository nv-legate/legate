/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

namespace logical_array_delinearize_test {

namespace {

using LogicalArrayDelinearizeUnit = DefaultFixture;

// delinearize test for bound fixed size array: primitive type array and struct type array
class DelinearizeFixedArrayTest
  : public LogicalArrayDelinearizeUnit,
    public ::testing::WithParamInterface<std::tuple<
      legate::Type,
      bool,
      std::tuple<std::int32_t, std::vector<std::uint64_t>, std::vector<std::uint64_t>>>> {};

// delinearize test for arrays not able to perform delinearize: unbound array, variable size array:
// list array and string array
class NonDelinearizeTest : public LogicalArrayDelinearizeUnit,
                           public ::testing::WithParamInterface<std::tuple<legate::Type, bool>> {};

// negative cases for delinearizing bound fixed size array and bound variable array
class NegativeDelinearizeTest
  : public LogicalArrayDelinearizeUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type, bool, std::tuple<std::int32_t, std::vector<std::uint64_t>>>> {};

class NegativeDelinearizeDimTest : public NegativeDelinearizeTest {};

class NegativeDelinearizeSizeTest : public NegativeDelinearizeTest {};

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayDelinearizeUnit,
  DelinearizeFixedArrayTest,
  ::testing::Combine(
    ::testing::Values(legate::int64(), logical_array_util_test::struct_type()),
    ::testing::Values(true, false),
    ::testing::Values(
      std::make_tuple(
        0, std::vector<std::uint64_t>({1, 1}), std::vector<std::uint64_t>({1, 1, 2, 3, 4})),
      std::make_tuple(1, std::vector<std::uint64_t>({2}), std::vector<std::uint64_t>({1, 2, 3, 4})),
      std::make_tuple(
        2, std::vector<std::uint64_t>({1, 3}), std::vector<std::uint64_t>({1, 2, 1, 3, 4})),
      std::make_tuple(3,
                      std::vector<std::uint64_t>({2, 1, 2, 1}),
                      std::vector<std::uint64_t>({1, 2, 3, 2, 1, 2, 1})))));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayDelinearizeUnit,
  NonDelinearizeTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayDelinearizeUnit,
  NegativeDelinearizeDimTest,
  ::testing::Combine(::testing::Values(legate::uint16(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(4, std::vector<std::uint64_t>({1, 1})),
                                       std::make_tuple(-1, std::vector<std::uint64_t>({1})))));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayDelinearizeUnit,
  NegativeDelinearizeSizeTest,
  ::testing::Combine(::testing::Values(legate::uint16(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(0, std::vector<std::uint64_t>({1, 2})),
                                       std::make_tuple(0, std::vector<std::uint64_t>({-1UL, 1})),
                                       std::make_tuple(3, std::vector<std::uint64_t>({2})),
                                       std::make_tuple(0,
                                                       std::vector<std::uint64_t>({-2UL, -2UL})))));

void test_negative_delinearize(const legate::Type& type,
                               bool bound,
                               bool nullable,
                               std::int32_t dim,
                               const std::vector<std::uint64_t>& sizes)
{
  auto array = logical_array_util_test::create_array_with_type(type, bound, nullable);

  if (type.variable_size()) {
    ASSERT_THROW(static_cast<void>(array.delinearize(dim, sizes)), std::runtime_error);
  } else {
    ASSERT_THROW(static_cast<void>(array.delinearize(dim, sizes)), std::invalid_argument);
  }
}

}  // namespace

TEST_P(DelinearizeFixedArrayTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, sizes, shape]      = params;

  auto bound_array  = logical_array_util_test::create_array_with_type(type, true, nullable);
  auto delinearized = bound_array.delinearize(dim, sizes);

  ASSERT_EQ(delinearized.extents().data(), shape);
}

TEST_P(NonDelinearizeTest, Basic)
{
  const auto [type, nullable] = GetParam();

  test_negative_delinearize(type, false, nullable, 0, {1});
}

TEST_P(NegativeDelinearizeDimTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, sizes]             = params;

  test_negative_delinearize(type, true, nullable, dim, sizes);
}

TEST_P(NegativeDelinearizeSizeTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [dim, sizes]             = params;

  test_negative_delinearize(type, true, nullable, dim, sizes);
}

}  // namespace logical_array_delinearize_test
