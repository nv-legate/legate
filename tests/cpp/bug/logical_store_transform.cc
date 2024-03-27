/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace logical_store_transform {

using LogicalStoreTransform = DefaultFixture;

TEST_F(LogicalStoreTransform, SliceBug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 3}, legate::int64());

  auto test_slice = [&store](auto&& slice) {
    EXPECT_EQ(slice.volume(), 0);
    EXPECT_FALSE(slice.transformed());
    EXPECT_EQ(slice.type(), store.type());
    EXPECT_FALSE(slice.overlaps(store));
    EXPECT_EQ(slice.dim(), store.dim());
  };
  test_slice(store.slice(1, legate::Slice(-9, -8)));
  test_slice(store.slice(1, legate::Slice(-8, -10)));
  test_slice(store.slice(1, legate::Slice(1, 1)));
}

TEST_F(LogicalStoreTransform, SliceBug2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 3}, legate::int64());

  auto test_slice = [&store](auto&& slice) {
    EXPECT_EQ(slice.volume(), 0);
    EXPECT_FALSE(slice.transformed());
    EXPECT_EQ(slice.type(), store.type());
    EXPECT_FALSE(slice.overlaps(store));
    EXPECT_EQ(slice.dim(), store.dim());
  };

  test_slice(store.slice(1, legate::Slice(-1, 0)));
  test_slice(store.slice(1, legate::Slice(-1, 1)));
  test_slice(store.slice(1, legate::Slice(10, 8)));
}

}  // namespace logical_store_transform
