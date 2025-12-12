/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_delinearize_unit {

namespace {

using LogicalStoreDelinearizeUnit = DefaultFixture;

constexpr std::uint64_t SCALAR_VALUE = 100;

class DelinearizeBoundStoreTest
  : public LogicalStoreDelinearizeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape,
                                                    std::int32_t,
                                                    std::vector<std::uint64_t>,
                                                    std::vector<std::uint64_t>>> {};

class DelinearizeScalarStoreTest
  : public LogicalStoreDelinearizeUnit,
    public ::testing::WithParamInterface<
      std::tuple<std::vector<std::uint64_t>, std::vector<std::uint64_t>>> {};

class NegativeDelinearizeStoreDimTest : public LogicalStoreDelinearizeUnit,
                                        public ::testing::WithParamInterface<std::int32_t> {};

class NegativeDelinearizeBoundStoreSizeTest
  : public LogicalStoreDelinearizeUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape, std::int32_t, std::vector<std::uint64_t>>> {};

class NegativeDelinearizeScalarStoreSizeTest
  : public LogicalStoreDelinearizeUnit,
    public ::testing::WithParamInterface<std::vector<std::uint64_t>> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreDelinearizeUnit,
  DelinearizeBoundStoreTest,
  ::testing::Values(std::make_tuple(legate::Shape{4, 3},
                                    0,
                                    std::vector<std::uint64_t>({4}),
                                    std::vector<std::uint64_t>({4, 3})),
                    std::make_tuple(legate::Shape{4, 3},
                                    0,
                                    std::vector<std::uint64_t>({2, 2}),
                                    std::vector<std::uint64_t>({2, 2, 3})),
                    std::make_tuple(legate::Shape{4, 3},
                                    0,
                                    std::vector<std::uint64_t>({1, 1, 4}),
                                    std::vector<std::uint64_t>({1, 1, 4, 3})),
                    std::make_tuple(legate::Shape{4, 1},
                                    0,
                                    std::vector<std::uint64_t>({4, 1, 1, 1, 1}),
                                    std::vector<std::uint64_t>({4, 1, 1, 1, 1, 1}))));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreDelinearizeUnit,
  DelinearizeScalarStoreTest,
  ::testing::Values(
    std::make_tuple(std::vector<std::uint64_t>({1}), std::vector<std::uint64_t>({1})),
    std::make_tuple(std::vector<std::uint64_t>({1, 1}), std::vector<std::uint64_t>({1, 1})),
    std::make_tuple(std::vector<std::uint64_t>({1, 1, 1, 1}),
                    std::vector<std::uint64_t>({1, 1, 1, 1})),
    std::make_tuple(std::vector<std::uint64_t>({1, 1, 1, 1, 1}),
                    std::vector<std::uint64_t>({1, 1, 1, 1, 1}))));

INSTANTIATE_TEST_SUITE_P(LogicalStoreDelinearizeUnit,
                         NegativeDelinearizeStoreDimTest,
                         ::testing::Values(-1, LEGATE_MAX_DIM));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreDelinearizeUnit,
  NegativeDelinearizeBoundStoreSizeTest,
  ::testing::Values(
    std::make_tuple(legate::Shape{4, 3}, 0, std::vector<std::uint64_t>({1})),
    std::make_tuple(legate::Shape{4, 3}, 1, std::vector<std::uint64_t>({1, 2})),
    std::make_tuple(legate::Shape{4, 3}, 0, std::vector<std::uint64_t>({1, 2, 3}))));

INSTANTIATE_TEST_SUITE_P(LogicalStoreDelinearizeUnit,
                         NegativeDelinearizeScalarStoreSizeTest,
                         ::testing::Values(std::vector<std::uint64_t>({2}),
                                           std::vector<std::uint64_t>({1, 2})));

}  // namespace

TEST_P(DelinearizeBoundStoreTest, Basic)
{
  const auto [shape, dim, sizes, delinearize_shape] = GetParam();
  auto runtime                                      = legate::Runtime::get_runtime();
  auto store                                        = runtime->create_store(shape, legate::int64());
  auto delinearize                                  = store.delinearize(dim, sizes);

  ASSERT_EQ(delinearize.extents().data(), delinearize_shape);
  ASSERT_TRUE(delinearize.transformed());
  ASSERT_EQ(delinearize.type(), store.type());
  ASSERT_TRUE(delinearize.overlaps(store));
  ASSERT_EQ(delinearize.dim(), delinearize_shape.size());
}

TEST_P(DelinearizeScalarStoreTest, Basic)
{
  const auto [sizes, delinearize_shape] = GetParam();
  auto runtime                          = legate::Runtime::get_runtime();
  auto store                            = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto delinearize                      = store.delinearize(/*dim=*/0, sizes);

  ASSERT_EQ(delinearize.extents().data(), delinearize_shape);
  ASSERT_TRUE(delinearize.transformed());
  ASSERT_EQ(delinearize.type(), store.type());
  ASSERT_TRUE(delinearize.overlaps(store));
  ASSERT_EQ(delinearize.dim(), delinearize_shape.size());
}

TEST_P(NegativeDelinearizeStoreDimTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{1}, legate::int64());

  ASSERT_THROW(static_cast<void>(store.delinearize(GetParam(), {1})), std::invalid_argument);
}

TEST_P(NegativeDelinearizeBoundStoreSizeTest, BoundStore)
{
  const auto [shape, dim, sizes] = GetParam();
  auto runtime                   = legate::Runtime::get_runtime();
  auto store                     = runtime->create_store(shape, legate::uint32());

  ASSERT_THROW(static_cast<void>(store.delinearize(dim, sizes)), std::invalid_argument);
}

TEST_P(NegativeDelinearizeStoreDimTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.delinearize(GetParam(), {1, 1})), std::invalid_argument);
}

TEST_P(NegativeDelinearizeScalarStoreSizeTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.delinearize(0, GetParam())), std::invalid_argument);
}

}  // namespace logical_store_delinearize_unit
