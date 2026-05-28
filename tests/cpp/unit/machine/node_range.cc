/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/hash.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace node_range_test {

using NodeRangeTest = DefaultFixture;

TEST_F(NodeRangeTest, ComparisonOperators)
{
  constexpr legate::mapping::NodeRange range1{1, 3};
  constexpr legate::mapping::NodeRange range2{2, 3};
  constexpr legate::mapping::NodeRange range3{1, 4};

  // Test NodeRange operators
  static_assert(range1 < range2);
  static_assert(range1 < range3);
  static_assert(range1 != range2);
  static_assert(range1 != range3);
  static_assert(!(range1 == range2));
  static_assert(!(range1 == range3));
}

TEST_F(NodeRangeTest, Hash)
{
  constexpr legate::mapping::NodeRange range{1, 3};
  constexpr legate::mapping::NodeRange same{1, 3};
  constexpr legate::mapping::NodeRange different_low{2, 3};
  constexpr legate::mapping::NodeRange different_high{1, 4};

  ASSERT_EQ(range.hash(), legate::hash_all(range.low, range.high));
  ASSERT_EQ(range.hash(), same.hash());
  ASSERT_NE(range.hash(), different_low.hash());
  ASSERT_NE(range.hash(), different_high.hash());
}

}  // namespace node_range_test
