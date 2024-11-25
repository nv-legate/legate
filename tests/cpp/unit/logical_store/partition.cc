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

#include <gtest/gtest.h>

namespace logical_store_partition_test {

namespace {

using LogicalStorePartitionUnit = DefaultFixture;

class PartitionTest
  : public LogicalStorePartitionUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape,
                 std::tuple<std::vector<std::uint64_t>, std::vector<std::uint64_t>>>> {};

class NegativePartitionTest : public LogicalStorePartitionUnit,
                              public ::testing::WithParamInterface<std::vector<std::uint64_t>> {};

class NegativePartitionSizeTest : public NegativePartitionTest {};

class NegativePartitionVolumeTest : public NegativePartitionTest {};

class UnboundStorePartitionTest : public NegativePartitionTest {};

class ChildStoreTest : public LogicalStorePartitionUnit,
                       public ::testing::WithParamInterface<
                         std::tuple<legate::Shape,
                                    std::vector<std::uint64_t>,
                                    std::tuple<legate::tuple<std::uint64_t>, legate::Shape>>> {};

class NegativeColorTest
  : public LogicalStorePartitionUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, legate::tuple<std::uint64_t>>> {
};

INSTANTIATE_TEST_SUITE_P(
  LogicalStorePartitionUnit,
  PartitionTest,
  ::testing::Combine(
    ::testing::Values(legate::Shape{7, 8}),
    ::testing::Values(
      std::make_tuple(std::vector<std::uint64_t>({2, 4}), std::vector<std::uint64_t>({4, 2})),
      std::make_tuple(std::vector<std::uint64_t>({5, 3}), std::vector<std::uint64_t>({2, 3})),
      std::make_tuple(std::vector<std::uint64_t>({1, 1}), std::vector<std::uint64_t>({7, 8})),
      std::make_tuple(std::vector<std::uint64_t>({7, 8}), std::vector<std::uint64_t>({1, 1})),
      std::make_tuple(std::vector<std::uint64_t>({8, 4}), std::vector<std::uint64_t>({1, 2})),
      std::make_tuple(std::vector<std::uint64_t>({9, 5}), std::vector<std::uint64_t>({1, 2})),
      std::make_tuple(std::vector<std::uint64_t>({2, 10}), std::vector<std::uint64_t>({4, 1})),
      std::make_tuple(std::vector<std::uint64_t>({10, 20}), std::vector<std::uint64_t>({1, 1})))));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         NegativePartitionSizeTest,
                         ::testing::Values(std::vector<std::uint64_t>({}),
                                           std::vector<std::uint64_t>({1, 2, 3})));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         NegativePartitionVolumeTest,
                         ::testing::Values(std::vector<std::uint64_t>({0}),
                                           std::vector<std::uint64_t>({1, 0}),
                                           std::vector<std::uint64_t>({1, 2, 0})));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         UnboundStorePartitionTest,
                         ::testing::Values(std::vector<std::uint64_t>({}),
                                           std::vector<std::uint64_t>({0})));

INSTANTIATE_TEST_SUITE_P(
  LogicalStorePartitionUnit,
  ChildStoreTest,
  ::testing::Combine(
    ::testing::Values(legate::Shape{9, 8}),
    ::testing::Values(std::vector<std::uint64_t>({2, 4})),
    ::testing::Values(std::make_tuple(legate::tuple<std::uint64_t>({0, 0}), legate::Shape{2, 4}),
                      std::make_tuple(legate::tuple<std::uint64_t>({2, 1}), legate::Shape{2, 4}),
                      std::make_tuple(legate::tuple<std::uint64_t>({4, 1}), legate::Shape{1, 4}))));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         NegativeColorTest,
                         ::testing::Combine(::testing::Values(legate::Shape{4, 8}),
                                            ::testing::Values(legate::tuple<std::uint64_t>({1}),
                                                              legate::tuple<std::uint64_t>({4, 2}),
                                                              legate::tuple<std::uint64_t>({5, 2}),
                                                              legate::tuple<std::uint64_t>({5,
                                                                                            0}))));

}  // namespace

TEST_P(PartitionTest, BoundStore)
{
  const auto [shape, params]           = GetParam();
  const auto [tile_shape, color_shape] = params;
  auto runtime                         = legate::Runtime::get_runtime();
  auto store                           = runtime->create_store(shape, legate::int64());
  auto partition                       = store.partition_by_tiling(tile_shape);

  ASSERT_EQ(partition.color_shape().data(), color_shape);
}

TEST_F(LogicalStorePartitionUnit, ScalarStore)
{
  auto runtime                         = legate::Runtime::get_runtime();
  constexpr std::uint64_t SCALAR_VALUE = 10;
  auto store                           = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto partition                       = store.partition_by_tiling({1});

  ASSERT_EQ(partition.color_shape().data(), std::vector<std::uint64_t>{1});
}

TEST_P(NegativePartitionSizeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 4}, legate::int64());

  // shape size mismatch
  ASSERT_THROW(static_cast<void>(store.partition_by_tiling(GetParam())), std::invalid_argument);
}

TEST_P(NegativePartitionVolumeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 4}, legate::int64());

  // volume is 0
  ASSERT_THROW(static_cast<void>(store.partition_by_tiling(GetParam())), std::invalid_argument);
}

TEST_P(NegativePartitionSizeTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{std::uint32_t{1}});

  // shape size mismatch
  ASSERT_THROW(static_cast<void>(store.partition_by_tiling(GetParam())), std::invalid_argument);
}

TEST_P(NegativePartitionVolumeTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{std::uint32_t{1}});

  // volume is 0
  ASSERT_THROW(static_cast<void>(store.partition_by_tiling(GetParam())), std::invalid_argument);
}

TEST_P(UnboundStorePartitionTest, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int64());

  ASSERT_THROW(static_cast<void>(store.partition_by_tiling(GetParam())), std::invalid_argument);
}

TEST_P(ChildStoreTest, Basic)
{
  const auto [shape, tile_shape, params] = GetParam();
  const auto [color, color_shape]        = params;
  auto runtime                           = legate::Runtime::get_runtime();
  auto store                             = runtime->create_store(shape, legate::int64());
  auto partition                         = store.partition_by_tiling(tile_shape);

  ASSERT_EQ(partition.store().shape(), store.shape());
  ASSERT_EQ(partition.get_child_store(color).shape(), color_shape);
}

TEST_P(NegativeColorTest, Basic)
{
  const auto [shape, color] = GetParam();
  auto runtime              = legate::Runtime::get_runtime();
  auto store                = runtime->create_store(shape, legate::int64());
  auto partition            = store.partition_by_tiling({1, 4});

  ASSERT_THROW(static_cast<void>(partition.get_child_store(color)), std::out_of_range);
}

}  // namespace logical_store_partition_test
