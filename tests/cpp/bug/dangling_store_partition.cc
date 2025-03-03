/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace dangling_store_partition {

using DanglingStorePartition = DefaultFixture;

namespace {

legate::LogicalStorePartition create_partition(const legate::Shape& shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(shape, legate::int64());
  return store.partition_by_tiling(store.shape().extents().data());
}

}  // namespace

TEST_F(DanglingStorePartition, Bug1)
{
  auto part1 = create_partition(legate::Shape{4, 2});
  auto part2 = create_partition(legate::Shape{4, 2});

  auto rf1 = part1.store().impl()->get_region_field();
  auto rf2 = part2.store().impl()->get_region_field();

  EXPECT_FALSE(rf1->region() == rf2->region() && rf1->field_id() == rf2->field_id());
}

}  // namespace dangling_store_partition
