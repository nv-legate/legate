/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <unit/logical_array/utils.h>
#include <utilities/utilities.h>

namespace logical_array_promote_test {

namespace {

using LogicalArrayPromoteUnit = DefaultFixture;

// promote test for bound fixed size array: primitive type array and struct type array
class PromoteFixedArrayTest
  : public LogicalArrayPromoteUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type,
                 bool,
                 std::tuple<std::int32_t, std::size_t, std::vector<std::uint64_t>>>> {};

// promote test for arrays not able to perform delinearize: unbound array, variable size array: list
// array and string array
class NonPromoteTest : public LogicalArrayPromoteUnit,
                       public ::testing::WithParamInterface<std::tuple<legate::Type, bool>> {};

// negative cases for promoting bound fixed size array and bound variable array
class NegativePromoteTest
  : public LogicalArrayPromoteUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Type, bool, std::tuple<std::int32_t, std::size_t>>> {};

class NegativePromoteDimTest : public NegativePromoteTest {};

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayPromoteUnit,
  PromoteFixedArrayTest,
  ::testing::Combine(
    ::testing::Values(legate::int64(), logical_array_util_test::struct_type()),
    ::testing::Values(true, false),
    ::testing::Values(std::make_tuple(0, 1, std::vector<std::uint64_t>({1, 1, 2, 3, 4})),
                      std::make_tuple(1, 5, std::vector<std::uint64_t>({1, 5, 2, 3, 4})),
                      std::make_tuple(2, 10, std::vector<std::uint64_t>({1, 2, 10, 3, 4})),
                      std::make_tuple(3, 15, std::vector<std::uint64_t>({1, 2, 3, 15, 4})),
                      std::make_tuple(4, 20, std::vector<std::uint64_t>({1, 2, 3, 4, 20})))));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayPromoteUnit,
  NonPromoteTest,
  ::testing::Combine(::testing::Values(legate::int32(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(
  LogicalArrayPromoteUnit,
  NegativePromoteDimTest,
  ::testing::Combine(::testing::Values(legate::uint16(),
                                       logical_array_util_test::struct_type(),
                                       logical_array_util_test::list_type(),
                                       legate::string_type()),
                     ::testing::Values(true, false),
                     ::testing::Values(std::make_tuple(-1, 1), std::make_tuple(5, 1))));

void test_negative_promote(
  const legate::Type& type, bool bound, bool nullable, std::int32_t extra_dim, std::size_t dim_size)
{
  auto array = logical_array_util_test::create_array_with_type(type, bound, nullable);

  if (type.variable_size()) {
    ASSERT_THROW(static_cast<void>(array.promote(extra_dim, dim_size)), std::runtime_error);
  } else {
    ASSERT_THROW(static_cast<void>(array.promote(extra_dim, dim_size)), std::invalid_argument);
  }
}

}  // namespace

TEST_P(PromoteFixedArrayTest, Basic)
{
  const auto [type, nullable, params]     = GetParam();
  const auto [extra_dim, dim_size, shape] = params;
  auto bound_array =
    logical_array_util_test::create_array_with_type(type, /*bound=*/true, nullable);
  auto promoted = bound_array.promote(extra_dim, dim_size);

  ASSERT_EQ(promoted.extents().data(), shape);
}

TEST_P(NonPromoteTest, Basic)
{
  const auto [type, nullable] = GetParam();

  test_negative_promote(type, /*bound=*/false, nullable, /*extra_dim=*/1, /*dim_size=*/1);
}

TEST_P(NegativePromoteDimTest, Basic)
{
  const auto [type, nullable, params] = GetParam();
  const auto [extra_dim, dim_size]    = params;

  test_negative_promote(type, /*bound=*/true, nullable, extra_dim, dim_size);
}

}  // namespace logical_array_promote_test
