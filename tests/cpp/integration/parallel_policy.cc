/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_parallel_policy {

using ParallelPolicyTest = DefaultFixture;

namespace {

constexpr std::size_t X_LEN       = 10;
constexpr std::size_t Y_LEN       = 10;
constexpr std::uint32_t OD_FACTOR = 4;
constexpr std::uint64_t THRESHOLD = 2;

}  // namespace

TEST_F(ParallelPolicyTest, PartitionSmallStore)
{
  auto* runtime    = legate::Runtime::get_runtime();
  const auto store = runtime->create_store(legate::Shape{X_LEN, Y_LEN}, legate::int32());

  auto pp = legate::ParallelPolicy{}
              .with_overdecompose_factor(OD_FACTOR)
              .with_streaming(legate::StreamingMode::RELAXED)
              .with_partitioning_threshold(legate::mapping::TaskTarget::CPU, THRESHOLD)
              .with_partitioning_threshold(legate::mapping::TaskTarget::GPU, THRESHOLD)
              .with_partitioning_threshold(legate::mapping::TaskTarget::OMP, THRESHOLD);

  {
    auto scope = legate::Scope{}.with_parallel_policy(pp);
    runtime->issue_fill(store, legate::Scalar{0});
  }

  auto part = store.get_partition();
  ASSERT_TRUE(part.has_value());
  if (part.has_value()) {
    auto color_shape = part->color_shape();
    ASSERT_THAT(color_shape, ::testing::Each(::testing::Gt(1)));
  }
}

}  // namespace test_parallel_policy
