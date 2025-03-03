/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_transpose_unit {

namespace {

using LogicalStoreTransposeUnit = DefaultFixture;

constexpr std::int32_t SCALAR_VALUE = 20;

class TransposeBoundStoreTest
  : public LogicalStoreTransposeUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape, std::vector<std::int32_t>, std::vector<std::uint64_t>>> {};

class NegativeTransposeBoundStoreTest
  : public LogicalStoreTransposeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, std::vector<std::int32_t>>> {};

class NegativeTransposeScalarStoreTest
  : public LogicalStoreTransposeUnit,
    public ::testing::WithParamInterface<std::vector<std::int32_t>> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreTransposeUnit,
  TransposeBoundStoreTest,
  ::testing::Values(std::make_tuple(legate::Shape{1},
                                    std::vector<std::int32_t>({0}),
                                    std::vector<std::uint64_t>({1})),
                    std::make_tuple(legate::Shape{1, 2},
                                    std::vector<std::int32_t>({1, 0}),
                                    std::vector<std::uint64_t>({2, 1})),
                    std::make_tuple(legate::Shape{1, 2, 3},
                                    std::vector<std::int32_t>({2, 0, 1}),
                                    std::vector<std::uint64_t>({3, 1, 2})),
                    std::make_tuple(legate::Shape{1, 2, 3, 4},
                                    std::vector<std::int32_t>({1, 2, 0, 3}),
                                    std::vector<std::uint64_t>({2, 3, 1, 4}))));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreTransposeUnit,
  NegativeTransposeBoundStoreTest,
  ::testing::Values(
    std::make_tuple(legate::Shape{1}, std::vector<std::int32_t>({})) /* invalid axes length */,
    std::make_tuple(legate::Shape{1}, std::vector<std::int32_t>({1, 0})) /* invalid axes length */,
    std::make_tuple(legate::Shape{1, 2}, std::vector<std::int32_t>({1})) /* invalid axes length */,
    std::make_tuple(legate::Shape{1, 2, 3},
                    std::vector<std::int32_t>({2, 2, 1})) /* axes has duplicates */,
    std::make_tuple(legate::Shape{1, 2, 3},
                    std::vector<std::int32_t>({-2, -2, 1})) /* axes has duplicates */,
    std::make_tuple(legate::Shape{1, 2, 3, 4},
                    std::vector<std::int32_t>({4, 0, 2, 1})) /* invalid axis in axes */,
    std::make_tuple(legate::Shape{1, 2, 3, 4},
                    std::vector<std::int32_t>({-3, 0, 2, 1})) /* invalid axis in axes */));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreTransposeUnit,
  NegativeTransposeScalarStoreTest,
  ::testing::Values(std::vector<std::int32_t>({}) /* invalid axes length */,
                    std::vector<std::int32_t>({0, 1}) /* invalid axes length */,
                    std::vector<std::int32_t>({1}) /* invalid axis in axes */,
                    std::vector<std::int32_t>({2, 2, 1}) /* invalid axis in axes */));
}  // namespace

TEST_P(TransposeBoundStoreTest, Basic)
{
  auto [shape, axes, transpose_shape] = GetParam();
  auto runtime                        = legate::Runtime::get_runtime();
  auto store                          = runtime->create_store(shape, legate::int64());
  auto transpose                      = store.transpose(std::move(axes));

  ASSERT_EQ(transpose.extents().data(), transpose_shape);
  ASSERT_TRUE(transpose.transformed());
  ASSERT_EQ(transpose.type(), store.type());
  ASSERT_TRUE(transpose.overlaps(store));
  ASSERT_EQ(transpose.dim(), store.dim());
}

TEST_F(LogicalStoreTransposeUnit, ScalarStore)
{
  auto runtime   = legate::Runtime::get_runtime();
  auto store     = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto transpose = store.transpose({0});

  ASSERT_EQ(transpose.extents().data(), std::vector<std::uint64_t>{1});
  ASSERT_TRUE(transpose.transformed());
  ASSERT_EQ(transpose.type(), store.type());
  ASSERT_TRUE(transpose.overlaps(store));
  ASSERT_EQ(transpose.dim(), store.dim());
}

TEST_P(NegativeTransposeBoundStoreTest, Basic)
{
  auto [shape, axes] = GetParam();
  auto runtime       = legate::Runtime::get_runtime();
  auto store         = runtime->create_store(shape, legate::int64());

  ASSERT_THROW(static_cast<void>(store.transpose(std::move(axes))), std::invalid_argument);
}

TEST_P(NegativeTransposeScalarStoreTest, Basic)
{
  auto axes    = GetParam();
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.transpose(std::move(axes))), std::invalid_argument);
}

TEST_F(LogicalStoreTransposeUnit, UnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int64(), 1);

  ASSERT_THROW(static_cast<void>(store.transpose({0})), std::invalid_argument);
}

TEST_F(LogicalStoreTransposeUnit, UnboundStoreInvalidDim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int64(), LEGATE_MAX_DIM);

  ASSERT_THROW(static_cast<void>(store.transpose({0})), std::invalid_argument);
}

}  // namespace logical_store_transpose_unit
