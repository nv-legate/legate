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

namespace {

class Square {
 public:
  template <class T>
  LEGATE_HOST_DEVICE T operator()(T x) const
  {
    return x * x;
  }
};

// NOLINTBEGIN(readability-magic-numbers, misc-const-correctness)

void test_elementwise_row_operation()
{
  auto input  = stl::create_store<std::int64_t>({3, 4});
  auto result = stl::create_store({3, 4}, std::int64_t{0});

  // Fill the input
  {
    auto input_view = stl::as_mdspan(input);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        input_view(i, j) = static_cast<std::int64_t>(i) * 4 + j;
      }
    }
  }

  auto input_rows  = stl::rows_of(input);
  auto result_rows = stl::rows_of(result);

  auto in_iter  = input_rows.begin();
  auto res_iter = result_rows.begin();

  auto result_view = stl::as_mdspan(result);
  EXPECT_EQ(result_view.rank(), 2);
  EXPECT_EQ(result_view.extent(0), 3);
  EXPECT_EQ(result_view.extent(1), 4);

  {
    auto res_row = *res_iter;
    stl::assign(stl::as_mdspan(res_row), stl::elementwise(Square{})(stl::as_mdspan(*in_iter)));
  }

  EXPECT_EQ(result_view(0, 0), 0);
  EXPECT_EQ(result_view(0, 1), 1);
  EXPECT_EQ(result_view(0, 2), 4);
  EXPECT_EQ(result_view(0, 3), 9);

  {
    ++in_iter;
    ++res_iter;
    auto res_row = *res_iter;
    stl::assign(stl::as_mdspan(res_row), stl::elementwise(Square{})(stl::as_mdspan(*in_iter)));
  }

  EXPECT_EQ(result_view(1, 0), 16);
  EXPECT_EQ(result_view(1, 1), 25);
  EXPECT_EQ(result_view(1, 2), 36);
  EXPECT_EQ(result_view(1, 3), 49);

  {
    ++in_iter;
    ++res_iter;
    auto res_row = *res_iter;
    stl::assign(stl::as_mdspan(res_row), stl::elementwise(Square{})(stl::as_mdspan(*in_iter)));
  }

  EXPECT_EQ(result_view(2, 0), 64);
  EXPECT_EQ(result_view(2, 1), 81);
  EXPECT_EQ(result_view(2, 2), 100);
  EXPECT_EQ(result_view(2, 3), 121);

  // make sure we haven't stomped any of the output
  EXPECT_EQ(result_view(1, 0), 16);
  EXPECT_EQ(result_view(1, 1), 25);
  EXPECT_EQ(result_view(1, 2), 36);
  EXPECT_EQ(result_view(1, 3), 49);

  EXPECT_EQ(result_view(0, 0), 0);
  EXPECT_EQ(result_view(0, 1), 1);
  EXPECT_EQ(result_view(0, 2), 4);
  EXPECT_EQ(result_view(0, 3), 9);
}

void elementwise_doxy_snippets()
{
  /// [elementwise example]
  // Perform element-wise addition of the rows of two logical stores,
  // assigning the result element-wise into the rows of the first.
  stl::logical_store<int, 2> store1 = {
    {1, 2, 3, 4},  // row 0
    {2, 3, 4, 5},  // row 1
    {3, 4, 5, 6}   // row 2
  };
  stl::logical_store<int, 2> store2 = {
    {10, 20, 30, 40},  // row 0
    {20, 30, 40, 50},  // row 1
    {30, 40, 50, 60}   // row 2
  };
  stl::transform(stl::rows_of(store1),
                 stl::rows_of(store2),
                 stl::rows_of(store1),
                 stl::elementwise(std::plus<>{}));

  // store1 now contains:
  // [[11 22 33 44]   // row 0
  //  [22 33 44 55]   // row 1
  //  [33 44 55 66]]  // row 2
  /// [elementwise example]

  auto sp = stl::as_mdspan(store1);
  EXPECT_EQ(sp(0, 0), 11);
  EXPECT_EQ(sp(0, 1), 22);
  EXPECT_EQ(sp(0, 2), 33);
  EXPECT_EQ(sp(0, 3), 44);
  EXPECT_EQ(sp(1, 0), 22);
  EXPECT_EQ(sp(1, 1), 33);
  EXPECT_EQ(sp(1, 2), 44);
  EXPECT_EQ(sp(1, 3), 55);
  EXPECT_EQ(sp(2, 0), 33);
  EXPECT_EQ(sp(2, 1), 44);
  EXPECT_EQ(sp(2, 2), 55);
  EXPECT_EQ(sp(2, 3), 66);
}

// NOLINTEND(readability-magic-numbers, misc-const-correctness)

}  // namespace

TEST_F(STL, TestElementwiseRowOperation) { test_elementwise_row_operation(); }

TEST_F(STL, ElementwiseDoxySnippets) { elementwise_doxy_snippets(); }
