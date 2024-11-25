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

namespace logical_store_overlaps_test {

namespace {

using LogicalStoreOverlapsUnit = DefaultFixture;

const legate::Shape& multi_dim_shape()
{
  static const auto shape = legate::Shape{4, 6, 8};

  return shape;
}

}  // namespace

TEST_F(LogicalStoreOverlapsUnit, OverlapsUnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int32(), 2);

  ASSERT_TRUE(store.overlaps(store));
  ASSERT_TRUE(store.overlaps(legate::LogicalStore{store}));

  auto other = runtime->create_store(legate::int64(), 1);

  ASSERT_FALSE(store.overlaps(other));
}

TEST_F(LogicalStoreOverlapsUnit, OverlapsSelf)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{3}, legate::int32());

  ASSERT_TRUE(store.overlaps(store));
}

TEST_F(LogicalStoreOverlapsUnit, OverlapsOptimizeScalar)
{
  auto runtime   = legate::Runtime::get_runtime();
  auto store     = runtime->create_store(legate::Shape{3}, legate::int32());
  auto optimized = runtime->create_store(legate::Shape{3}, legate::int32(), true);

  ASSERT_FALSE(store.overlaps(optimized));
}

TEST_F(LogicalStoreOverlapsUnit, OverlapsSameRoot)
{
  auto runtime          = legate::Runtime::get_runtime();
  auto store_multi_dims = runtime->create_store(multi_dim_shape(), legate::int32());
  auto sliced           = store_multi_dims.slice(1, legate::Slice{1, 2});

  ASSERT_TRUE(store_multi_dims.overlaps(sliced));
  ASSERT_TRUE(sliced.overlaps(store_multi_dims));
}

TEST_F(LogicalStoreOverlapsUnit, OverlapsDifferentRoot)
{
  auto runtime          = legate::Runtime::get_runtime();
  auto store            = runtime->create_store(legate::Shape{3}, legate::int32());
  auto optimized        = runtime->create_store(legate::Shape{3}, legate::int32(), true);
  auto store_multi_dims = runtime->create_store(multi_dim_shape(), legate::int32());
  auto sliced           = store_multi_dims.slice(1, legate::Slice{1, 2});

  ASSERT_FALSE(store_multi_dims.overlaps(store));
  ASSERT_FALSE(store_multi_dims.overlaps(optimized));
  ASSERT_FALSE(sliced.overlaps(store));
  ASSERT_FALSE(sliced.overlaps(optimized));
}

TEST_F(LogicalStoreOverlapsUnit, OverlapsEmptyShape)
{
  auto runtime     = legate::Runtime::get_runtime();
  auto store       = runtime->create_store(legate::Shape{3}, legate::int32());
  auto empty_store = runtime->create_store(legate::Shape{}, legate::int32());

  ASSERT_FALSE(empty_store.overlaps(store));
  ASSERT_FALSE(store.overlaps(empty_store));
}

TEST_F(LogicalStoreOverlapsUnit, OverlapsScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{std::int64_t{1}});
  auto other   = runtime->create_store(legate::Scalar{std::int64_t{2}});

  ASSERT_TRUE(store.overlaps(store));
  ASSERT_TRUE(store.overlaps(legate::LogicalStore{store}));
  ASSERT_FALSE(store.overlaps(legate::LogicalStore{other}));
  ASSERT_FALSE(store.overlaps(other));
}

}  // namespace logical_store_overlaps_test
