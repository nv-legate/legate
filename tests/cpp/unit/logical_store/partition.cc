/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

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

class ColorShapeDimensionSizeTest
  : public LogicalStorePartitionUnit,
    public ::testing::WithParamInterface<std::vector<std::uint64_t>> {};

class ChildStoreTest : public LogicalStorePartitionUnit,
                       public ::testing::WithParamInterface<
                         std::tuple<legate::Shape,
                                    std::vector<std::uint64_t>,
                                    std::tuple<legate::tuple<std::uint64_t>, legate::Shape>>> {};

class ForcedColorChildStoreTest
  : public LogicalStorePartitionUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape,
                 std::vector<std::uint64_t>,
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

// For BoundStore (2D): tile_shape must be 2-tuple with volume = 0
INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         NegativePartitionVolumeTest,
                         ::testing::Values(std::vector<std::uint64_t>({0, 1}),
                                           std::vector<std::uint64_t>({1, 0}),
                                           std::vector<std::uint64_t>({0, 0})));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         UnboundStorePartitionTest,
                         ::testing::Values(std::vector<std::uint64_t>({1}),
                                           std::vector<std::uint64_t>({2}),
                                           std::vector<std::uint64_t>({1, 2})));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         ColorShapeDimensionSizeTest,
                         ::testing::Values(std::vector<std::uint64_t>({}),
                                           std::vector<std::uint64_t>({1, 2, 3})));

class ColorShapeZeroVolumeTest : public LogicalStorePartitionUnit,
                                 public ::testing::WithParamInterface<std::vector<std::uint64_t>> {
};

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         ColorShapeZeroVolumeTest,
                         ::testing::Values(std::vector<std::uint64_t>({0, 1}),
                                           std::vector<std::uint64_t>({1, 0}),
                                           std::vector<std::uint64_t>({0, 0})));

INSTANTIATE_TEST_SUITE_P(
  LogicalStorePartitionUnit,
  ChildStoreTest,
  ::testing::Combine(
    ::testing::Values(legate::Shape{9, 8}),
    ::testing::Values(std::vector<std::uint64_t>({2, 4})),
    ::testing::Values(std::make_tuple(legate::tuple<std::uint64_t>{{0, 0}}, legate::Shape{2, 4}),
                      std::make_tuple(legate::tuple<std::uint64_t>{{2, 1}}, legate::Shape{2, 4}),
                      std::make_tuple(legate::tuple<std::uint64_t>{{4, 1}}, legate::Shape{1, 4}))));

INSTANTIATE_TEST_SUITE_P(
  LogicalStorePartitionUnit,
  ForcedColorChildStoreTest,
  ::testing::Combine(
    ::testing::Values(legate::Shape{6, 9}),
    ::testing::Values(std::vector<std::uint64_t>({2, 4})),
    ::testing::Values(std::vector<std::uint64_t>({2, 5})),
    ::testing::Values(std::make_tuple(legate::tuple<std::uint64_t>{{0, 0}}, legate::Shape{2, 4}),
                      std::make_tuple(legate::tuple<std::uint64_t>{{1, 2}}, legate::Shape{2, 1}),
                      std::make_tuple(legate::tuple<std::uint64_t>{{0, 4}}, legate::Shape{2, 0}),
                      std::make_tuple(legate::tuple<std::uint64_t>{{1, 3}}, legate::Shape{2, 0}))));

INSTANTIATE_TEST_SUITE_P(LogicalStorePartitionUnit,
                         NegativeColorTest,
                         ::testing::Combine(::testing::Values(legate::Shape{4, 8}),
                                            ::testing::Values(legate::tuple<std::uint64_t>{1},
                                                              legate::tuple<std::uint64_t>{{4, 2}},
                                                              legate::tuple<std::uint64_t>{{5, 2}},
                                                              legate::tuple<std::uint64_t>{
                                                                {5, 0}})));

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

TEST_P(PartitionTest, ForcedColor)
{
  const auto [shape, params]           = GetParam();
  const auto [tile_shape, color_shape] = params;
  auto runtime                         = legate::Runtime::get_runtime();
  auto store                           = runtime->create_store(shape, legate::int64());
  auto partition                       = store.partition_by_tiling(tile_shape, color_shape);

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

TEST_F(LogicalStorePartitionUnit, ScalarStoreCreateProjection)
{
  auto runtime                         = legate::Runtime::get_runtime();
  constexpr std::uint64_t SCALAR_VALUE = 10;
  auto store                           = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  const auto& internal_store           = store.impl();
  auto no_partition                    = legate::detail::create_no_partition();

  // Create partition for scalar store
  auto store_partition = legate::detail::create_store_partition(internal_store, no_partition);

  // create_store_projection for scalar store should return empty projection
  auto projection = store_partition->create_store_projection(Legion::Domain{});

  // Empty StoreProjection has no partition
  ASSERT_FALSE(projection.partition.exists());
}

TEST_F(LogicalStorePartitionUnit, GetPlacementInfo)
{
  auto runtime   = legate::Runtime::get_runtime();
  auto store     = runtime->create_store(legate::Shape{8, 8}, legate::int64());
  auto partition = store.partition_by_tiling({2, 2});

  // Call get_placement_info to cover the interface
  auto placement_info = partition.get_placement_info();

  // Verify placements is not empty
  ASSERT_GT(placement_info.placements().size(), 0);
}

TEST_P(NegativePartitionSizeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 4}, legate::int64());

  // shape size mismatch
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(GetParam())); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Incompatible tile shape: expected a 2-tuple, got a")));
}

TEST_P(NegativePartitionSizeTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{std::uint32_t{1}});

  // shape size mismatch
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(GetParam())); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Incompatible tile shape: expected a 1-tuple, got a")));
}

TEST_P(NegativePartitionVolumeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 4}, legate::int64());

  // volume is 0
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(GetParam())); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Tile shape must have a volume greater than 0")));
}

TEST_F(LogicalStorePartitionUnit, ScalarStoreVolumeZero)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{std::uint32_t{1}});

  // Scalar store is 1D, so tile_shape must be 1-tuple with volume = 0
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling({0})); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Tile shape must have a volume greater than 0")));
}

TEST_P(ColorShapeDimensionSizeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 4}, legate::int64());
  std::vector<std::uint64_t> tile_shape{2, 2};
  const auto& color_shape = GetParam();

  // shape size mismatch
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(tile_shape, color_shape)); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Incompatible color shape: expected a 2-tuple, got a")));
}

TEST_P(ColorShapeDimensionSizeTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{std::uint32_t{1}});
  std::vector<std::uint64_t> tile_shape{1};
  const auto& color_shape = GetParam();

  // shape size mismatch
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(tile_shape, color_shape)); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Incompatible color shape: expected a 1-tuple, got a")));
}

TEST_P(ColorShapeZeroVolumeTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 4}, legate::int64());
  std::vector<std::uint64_t> tile_shape{2, 2};
  const auto& color_shape = GetParam();

  // color shape volume is 0
  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(tile_shape, color_shape)); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Color shape must have a volume greater than 0")));
}

TEST_P(UnboundStorePartitionTest, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  // Create unbound store with matching dimension
  const auto dim = static_cast<std::uint32_t>(GetParam().size());
  auto store     = runtime->create_store(legate::int64(), dim);

  ASSERT_THAT([&]() { static_cast<void>(store.partition_by_tiling(GetParam())); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
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

TEST_P(ForcedColorChildStoreTest, Basic)
{
  const auto [shape, tile_shape, forced_color, params] = GetParam();
  const auto [color, color_shape]                      = params;
  auto runtime                                         = legate::Runtime::get_runtime();
  auto store     = runtime->create_store(shape, legate::int64());
  auto partition = store.partition_by_tiling(tile_shape, forced_color);

  ASSERT_EQ(partition.color_shape().data(), forced_color);
  ASSERT_EQ(partition.store().shape(), store.shape());
  ASSERT_EQ(partition.get_child_store(color).shape(), color_shape);
}

TEST_P(NegativeColorTest, Basic)
{
  const auto& param      = GetParam();
  const auto& shape      = std::get<0>(param);
  const auto& test_color = std::get<1>(param);
  auto runtime           = legate::Runtime::get_runtime();
  auto store             = runtime->create_store(shape, legate::int64());
  auto partition         = store.partition_by_tiling({1, 4});

  ASSERT_THAT([&]() { static_cast<void>(partition.get_child_store(test_color)); },
              testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("is invalid for partition of color shape")));
}

TEST_F(LogicalStorePartitionUnit, UnboundStoreCreatePartition)
{
  auto runtime               = legate::Runtime::get_runtime();
  auto store                 = runtime->create_store(legate::int64());
  auto no_partition          = legate::detail::create_no_partition();
  const auto& internal_store = store.impl();

  ASSERT_THAT(
    [&] {
      static_cast<void>(legate::detail::create_store_partition(internal_store, no_partition));
    },
    testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Unbound store cannot be manually partitioned")));
}

TEST_F(LogicalStorePartitionUnit, GetChildStoreFromNonTilingPartition)
{
  auto runtime               = legate::Runtime::get_runtime();
  auto store                 = runtime->create_store(legate::Shape{4, 4}, legate::int64());
  auto no_partition          = legate::detail::create_no_partition();
  const auto& internal_store = store.impl();

  // Create a LogicalStorePartition with NoPartition (not Tiling)
  auto store_partition = legate::detail::create_store_partition(internal_store, no_partition);

  // get_child_store only supports Tiling partitions
  ASSERT_THAT([&] { static_cast<void>(store_partition->get_child_store({0, 0})); },
              testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Child stores can be retrieved only from tile partitions")));
}

}  // namespace logical_store_partition_test
