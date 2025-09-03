/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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

}  // namespace node_range_test
