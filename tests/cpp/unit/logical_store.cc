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

#include <gtest/gtest.h>

#include "legate.h"

namespace unit {

TEST(Store, Creation)
{
  // Bound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store({4, 4}, legate::int64());
    EXPECT_FALSE(store.unbound());
    EXPECT_EQ(store.dim(), 2);
    EXPECT_EQ(store.extents(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(store.type(), legate::int64());
    EXPECT_FALSE(store.transformed());
  }

  // Unbound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store(legate::int64());
    EXPECT_TRUE(store.unbound());
    EXPECT_EQ(store.dim(), 1);
    EXPECT_EQ(store.type(), legate::int64());
    EXPECT_FALSE(store.transformed());
    EXPECT_THROW(store.extents(), std::invalid_argument);
  }
}

TEST(Store, Transform)
{
  // Bound
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({4, 3}, legate::int64());

  auto promoted = store.promote(0, 5);
  EXPECT_EQ(promoted.extents(), (std::vector<size_t>{5, 4, 3}));
  EXPECT_TRUE(promoted.transformed());

  auto projected = store.project(0, 1);
  EXPECT_EQ(projected.extents(),
            (std::vector<size_t>{
              3,
            }));
  EXPECT_TRUE(projected.transformed());

  auto sliced = store.slice(1, legate::Slice(1, 3));
  EXPECT_EQ(sliced.extents(), (std::vector<size_t>{4, 2}));
  EXPECT_TRUE(sliced.transformed());

  auto transposed = store.transpose({1, 0});
  EXPECT_EQ(transposed.extents(), (std::vector<size_t>{3, 4}));
  EXPECT_TRUE(transposed.transformed());

  auto delinearized = store.delinearize(0, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(delinearized.extents(), (std::vector<size_t>{2, 2, 3}));
  EXPECT_TRUE(delinearized.transformed());
}

TEST(Store, InvalidTransform)
{
  // Bound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store({4, 3}, legate::int64());

    EXPECT_THROW(store.promote(3, 5), std::invalid_argument);
    EXPECT_THROW(store.promote(-3, 5), std::invalid_argument);

    EXPECT_THROW(store.project(2, 1), std::invalid_argument);
    EXPECT_THROW(store.project(-3, 1), std::invalid_argument);
    EXPECT_THROW(store.project(0, 4), std::invalid_argument);

    EXPECT_THROW(store.slice(2, legate::Slice(1, 3)), std::invalid_argument);

    EXPECT_THROW(store.transpose({
                   2,
                 }),
                 std::invalid_argument);
    EXPECT_THROW(store.transpose({0, 0}), std::invalid_argument);
    EXPECT_THROW(store.transpose({2, 0}), std::invalid_argument);

    EXPECT_THROW(store.delinearize(2, {2, 3}), std::invalid_argument);
    EXPECT_THROW(store.delinearize(0, {2, 3}), std::invalid_argument);
  }

  // Unbound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store(legate::int64());
    EXPECT_THROW(store.promote(1, 1), std::invalid_argument);
  }
}

}  // namespace unit
