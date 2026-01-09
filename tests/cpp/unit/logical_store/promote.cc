/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_promote_unit {

namespace {

using LogicalStorePromoteUnit = DefaultFixture;

constexpr std::int32_t SCALAR_VALUE = 10;

class PromoteBoundStoreTest
  : public LogicalStorePromoteUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape, std::int32_t, std::size_t, std::vector<std::uint64_t>>> {};

class PromoteScalarStoreTest
  : public LogicalStorePromoteUnit,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t, std::size_t, std::vector<std::uint64_t>>> {};

class NegativePromoteStoreDimTest : public LogicalStorePromoteUnit,
                                    public ::testing::WithParamInterface<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalStorePromoteUnit,
  PromoteBoundStoreTest,
  ::testing::Values(
    std::make_tuple(legate::Shape{4}, 0, 0, std::vector<std::uint64_t>({0, 4})),
    std::make_tuple(legate::Shape{3, 4}, 1, 1, std::vector<std::uint64_t>({3, 1, 4})),
    std::make_tuple(legate::Shape{2, 3, 4},
                    2,
                    -1,
                    std::vector<std::uint64_t>({2, 3, static_cast<std::uint64_t>(-1), 4})),
    std::make_tuple(legate::Shape{1, 2, 3, 4}, 4, 5, std::vector<std::uint64_t>({1, 2, 3, 4, 5}))));

INSTANTIATE_TEST_SUITE_P(
  LogicalStorePromoteUnit,
  PromoteScalarStoreTest,
  ::testing::Values(
    std::make_tuple(0, 5, std::vector<std::uint64_t>({5, 1})),
    std::make_tuple(1, -5, std::vector<std::uint64_t>({1, static_cast<std::uint64_t>(-5)}))));

INSTANTIATE_TEST_SUITE_P(LogicalStorePromoteUnit,
                         NegativePromoteStoreDimTest,
                         ::testing::Values(-1, LEGATE_MAX_DIM));

}  // namespace

TEST_P(PromoteBoundStoreTest, Basic)
{
  const auto [shape, extra_dim, dim_size, promote_shape] = GetParam();
  auto runtime                                           = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(shape, legate::int64());
  auto promote = store.promote(extra_dim, dim_size);

  ASSERT_EQ(promote.extents().data(), promote_shape);
  ASSERT_TRUE(promote.transformed());
  ASSERT_EQ(promote.type(), store.type());
  ASSERT_TRUE(promote.overlaps(store));
  ASSERT_EQ(promote.dim(), store.dim() + 1);
}

TEST_P(PromoteScalarStoreTest, Basic)
{
  const auto [extra_dim, dim_size, promote_shape] = GetParam();
  auto runtime                                    = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto promote = store.promote(extra_dim, dim_size);

  ASSERT_EQ(promote.extents().data(), promote_shape);
  ASSERT_TRUE(promote.transformed());
  ASSERT_EQ(promote.type(), store.type());
  ASSERT_TRUE(promote.overlaps(store));
  ASSERT_EQ(promote.dim(), store.dim() + 1);
}

TEST_P(NegativePromoteStoreDimTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 3}, legate::int64());

  ASSERT_THROW(static_cast<void>(store.promote(GetParam(), 1)), std::invalid_argument);
}

TEST_P(NegativePromoteStoreDimTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.promote(GetParam(), 2)), std::invalid_argument);
}

TEST_F(LogicalStorePromoteUnit, UnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int64(), LEGATE_MAX_DIM);

  ASSERT_THROW(static_cast<void>(store.promote(0, 1)), std::invalid_argument);
}

}  // namespace logical_store_promote_unit
