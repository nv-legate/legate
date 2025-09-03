/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// #include <legate.h>

// #include <legate/mapping/machine.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace processor_range_test {

using ProcessorRangeTest = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

TEST_F(ProcessorRangeTest, Create)
{
  constexpr legate::mapping::ProcessorRange range{1, 3, 0};

  static_assert(!range.empty());
  static_assert(range.per_node_count == 1);
  static_assert(range.low == 1);
  static_assert(range.high == 3);
  static_assert(range.count() == 2);
  static_assert(range.get_node_range() == legate::mapping::NodeRange{1, 3});
}

TEST_F(ProcessorRangeTest, CreateDefault)
{
  constexpr legate::mapping::ProcessorRange range;

  static_assert(range.empty());
  static_assert(range.per_node_count == 1);
  static_assert(range.low == 0);
  static_assert(range.high == 0);
  static_assert(range.count() == 0);
}

TEST_F(ProcessorRangeTest, CreateEmpty)
{
  constexpr auto check_empty = [](const legate::mapping::ProcessorRange& range) {
    ASSERT_TRUE(range.empty());
    ASSERT_EQ(range.per_node_count, 1);
    ASSERT_EQ(range.low, 0);
    ASSERT_EQ(range.high, 0);
    ASSERT_EQ(range.count(), 0);
    ASSERT_THROW(static_cast<void>(range.get_node_range()), std::invalid_argument);
  };

  constexpr legate::mapping::ProcessorRange range1{1, 0, 1};
  check_empty(range1);

  constexpr legate::mapping::ProcessorRange range2{3, 3, 0};
  check_empty(range2);
}

TEST_F(ProcessorRangeTest, ComparisonOperator)
{
  constexpr legate::mapping::ProcessorRange range1{2, 6, 2};
  constexpr legate::mapping::ProcessorRange range2{2, 6, 2};
  static_assert(range1 == range2);
  static_assert(!(range1 < range2));
  static_assert(!(range2 < range1));

  constexpr legate::mapping::ProcessorRange range3{1, 6, 2};
  static_assert(!(range1 == range3));
  static_assert(range1 != range3);
  static_assert(range3 < range1);
  static_assert(!(range1 < range3));

  constexpr legate::mapping::ProcessorRange range4{2, 5, 2};
  static_assert(!(range1 == range4));
  static_assert(range1 != range4);
  static_assert(range4 < range1);
  static_assert(!(range1 < range4));

  constexpr legate::mapping::ProcessorRange range5{2, 6, 1};
  static_assert(!(range1 == range5));
  static_assert(range1 != range5);
  static_assert(range5 < range1);
  static_assert(!(range1 < range5));
}

TEST_F(ProcessorRangeTest, IntersectionOperator)
{
  // Generate nonempty range
  constexpr legate::mapping::ProcessorRange range1{0, 3, 1};
  constexpr legate::mapping::ProcessorRange range2{2, 4, 1};
  constexpr auto result1 = range1 & range2;
  static_assert(result1 == legate::mapping::ProcessorRange{2, 3, 1});

  constexpr legate::mapping::ProcessorRange range3{2, 4, 1};
  constexpr legate::mapping::ProcessorRange range4{1, 3, 1};
  constexpr auto result2 = range3 & range4;
  static_assert(result2 == legate::mapping::ProcessorRange{2, 3, 1});

  // Generate empty range
  constexpr legate::mapping::ProcessorRange range5{0, 2, 1};
  constexpr legate::mapping::ProcessorRange range6{3, 5, 1};
  constexpr auto result3 = range5 & range6;
  static_assert(result3 == legate::mapping::ProcessorRange{0, 0, 1});
  static_assert(result3.count() == 0);
}

TEST_F(ProcessorRangeTest, NegativeIntersectionOperator)
{
  constexpr legate::mapping::ProcessorRange range1{1, 3, 1};
  constexpr legate::mapping::ProcessorRange range2{2, 4, 2};
  ASSERT_THROW(static_cast<void>(range1 & range2), std::invalid_argument);
}

TEST_F(ProcessorRangeTest, NodeRange)
{
  constexpr legate::mapping::ProcessorRange range{0, 7, 2};
  static_assert(range.get_node_range() == legate::mapping::NodeRange{0, 4});
}

TEST_F(ProcessorRangeTest, Slice)
{
  // Slice empty range with empty range
  constexpr legate::mapping::ProcessorRange range1{3, 1, 1};
  static_assert(range1.slice(0, 0).count() == 0);
  static_assert(range1.slice(4, 6).count() == 0);

  // Slice nonempty range with empty range
  constexpr legate::mapping::ProcessorRange range2{1, 3, 1};
  static_assert(range2.slice(0, 0).count() == 0);
  static_assert(range2.slice(4, 6).count() == 0);
  static_assert(range2.slice(1, 0).count() == 0);

  // Slice nonempty range with nonempty range
  constexpr legate::mapping::ProcessorRange range3{1, 3, 1};
  static_assert(range3.slice(1, 3).count() == 1);
  static_assert(range3.slice(0, 2).count() == 2);
}

TEST_F(ProcessorRangeTest, ToString)
{
  constexpr legate::mapping::ProcessorRange range{1, 3, 1};
  constexpr std::string_view range_str = "Proc([1,3], 1 per node)";

  std::stringstream ss;
  ss << range;
  ASSERT_EQ(ss.str(), range_str);
  ASSERT_EQ(range.to_string(), range_str);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace processor_range_test
