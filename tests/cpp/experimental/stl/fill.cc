/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/stl.hpp>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

using STL = DefaultFixture;

namespace stl = legate::experimental::stl;

// NOLINTBEGIN(readability-magic-numbers, misc-const-correctness)

TEST_F(STL, Test1DFill)
{
  auto store = stl::create_store<std::int64_t>({4});

  stl::fill(store, /*val=*/1);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 1);
  }

  stl::fill(store, /*val=*/2);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 2);
  }
}

TEST_F(STL, Test2DFill)
{
  auto store = stl::create_store<std::int64_t>({4, 5});

  stl::fill(store, /*val=*/1);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 1);
  }

  stl::fill(store, /*val=*/2);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 2);
  }
}

TEST_F(STL, Test3DFill)
{
  auto store = stl::create_store<std::int64_t>({4, 5, 6});

  stl::fill(store, /*val=*/1);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 1);
  }

  stl::fill(store, /*val=*/2);

  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 2);
  }
}

// Test that we can fill a slice of a store
TEST_F(STL, TestFillSlice)
{
  auto store = stl::create_store<std::int64_t>({2, 2}, /*value=*/1);

  // Fill the first row with 2
  stl::fill(*stl::rows_of(store).begin(), /*val=*/2);

  auto span = stl::as_mdspan(store);
  EXPECT_EQ(span(0, 0), 2);
  EXPECT_EQ(span(0, 1), 2);
  EXPECT_EQ(span(1, 0), 1);
  EXPECT_EQ(span(1, 1), 1);
}

TEST_F(STL, FillDoxySnippets)
{
  /// [fill example]
  // Declare a 3-dimensional logical store and fill it with the value 42.
  stl::logical_store<int, 3> store{{100, 200, 300}};
  stl::fill(store, /*val=*/42);
  // store's elements are now all 42
  /// [fill example]

  auto sp = stl::as_mdspan(store);
  EXPECT_EQ(sp(0, 0, 0), 42);
  EXPECT_EQ(sp(99, 200, 299), 42);
}

// NOLINTEND(readability-magic-numbers, misc-const-correctness)
