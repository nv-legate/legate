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

#include "stl/stl.hpp"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

using STL = LegateSTLFixture;

namespace stl = legate::stl;

TEST_F(STL, Test1DFill)
{
  auto store = stl::create_store<std::int64_t>({4});

  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 0);
  }

  stl::fill(store, 1);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 1);
  }

  stl::fill(store, 2);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 2);
  }
}

TEST_F(STL, Test2DFill)
{
  auto store = stl::create_store<std::int64_t>({4, 5});

  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 0);
  }

  stl::fill(store, 1);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 1);
  }

  stl::fill(store, 2);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 2);
  }
}

TEST_F(STL, Test3DFill)
{
  auto store = stl::create_store<std::int64_t>({4, 5, 6});

  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 0);
  }

  stl::fill(store, 1);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 1);
  }

  stl::fill(store, 2);

  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 2);
  }
}

// Test that we can fill a slice of a store
TEST_F(STL, TestFillSlice)
{
  auto store = stl::create_store<std::int64_t>({2, 2}, 1);

  // Fill the first row with 2
  stl::fill(*stl::rows_of(store).begin(), 2);

  auto span = stl::as_mdspan(store);
  EXPECT_EQ(span(0, 0), 2);
  EXPECT_EQ(span(0, 1), 2);
  EXPECT_EQ(span(1, 0), 1);
  EXPECT_EQ(span(1, 1), 1);
}
