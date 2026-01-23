/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_broadcast_unit {

namespace {

using LogicalStoreBroadcastUnit = DefaultFixture;

}  // namespace

TEST_F(LogicalStoreBroadcastUnit, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  // Create a store with shape {1, 4} - dimension 0 has size 1
  auto store = runtime->create_store(legate::Shape{1, 4}, legate::int64());

  // Broadcast dimension 0 from size 1 to size 3
  auto broadcasted = store.broadcast(/*dim=*/0, /*dim_size=*/3);

  ASSERT_EQ(broadcasted.extents().data(), (std::vector<std::uint64_t>{3, 4}));
  ASSERT_TRUE(broadcasted.transformed());
  ASSERT_EQ(broadcasted.type(), store.type());
  ASSERT_TRUE(broadcasted.overlaps(store));
  ASSERT_EQ(broadcasted.dim(), store.dim());
}

TEST_F(LogicalStoreBroadcastUnit, InvalidDimension)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{1, 4}, legate::int64());

  // Dimension out of range should throw
  ASSERT_THAT([&] { static_cast<void>(store.broadcast(/*dim=*/-1, /*dim_size=*/3)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid broadcast on dimension")));

  ASSERT_THAT([&] { static_cast<void>(store.broadcast(/*dim=*/2, /*dim_size=*/3)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid broadcast on dimension")));
}

TEST_F(LogicalStoreBroadcastUnit, DimensionSizeNotOne)
{
  auto runtime = legate::Runtime::get_runtime();
  // Create a store with shape {4, 3} - neither dimension has size 1
  auto store = runtime->create_store(legate::Shape{4, 3}, legate::int64());

  // Trying to broadcast dimension 0 (size 4, not 1) should throw
  ASSERT_THAT([&] { static_cast<void>(store.broadcast(/*dim=*/0, /*dim_size=*/5)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("expected size 1 but got")));

  // Trying to broadcast dimension 1 (size 3, not 1) should throw
  ASSERT_THAT([&] { static_cast<void>(store.broadcast(/*dim=*/1, /*dim_size=*/5)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("expected size 1 but got")));
}

}  // namespace logical_store_broadcast_unit
