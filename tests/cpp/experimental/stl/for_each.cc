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

#include "legate/experimental/stl.hpp"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

using STL = DefaultFixture;

namespace stl = legate::experimental::stl;

// NOLINTBEGIN(readability-magic-numbers, misc-const-correctness)

namespace {

// clang-tidy complains that "1D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void for_each_1D()
{
  stl::logical_store<std::int64_t, 1> store = {std::in_place,  //
                                               {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  stl::for_each(store, [] LEGATE_HOST_DEVICE(std::int64_t & x) { ++x; });

  // Check the result
  std::int64_t expected = 0;
  for (auto actual : stl::elements_of(store)) {
    EXPECT_EQ(actual, ++expected);
  }
}

void for_each_zip_doxy_snippets()
{
  {
    /// [stl-for-each-zip-elements]
    // Element-wise addition of two logical stores.
    stl::logical_store<int, 2> store1 = {{1, 2, 3, 4},  //
                                         {2, 3, 4, 5},
                                         {3, 4, 5, 6}};
    stl::logical_store<int, 2> store2 = {{3, 4, 5, 6},  //
                                         {2, 3, 4, 5},
                                         {1, 2, 3, 4}};

    // `a` and `b` refer to the elements of `store1` and `store2`.
    auto fn = [] LEGATE_HOST_DEVICE(int& a, int& b) { a += b; };
    stl::for_each_zip(fn, store1, store2);

    // store1 now contains:
    // {
    //   {4, 6, 8, 10},
    //   {4, 6, 8, 10},
    //   {4, 6, 8, 10}
    // }
    /// [stl-for-each-zip-elements]

    auto sp = stl::as_mdspan(store1);
    EXPECT_EQ(sp(0, 0), 4);
    EXPECT_EQ(sp(0, 1), 6);
    EXPECT_EQ(sp(0, 2), 8);
    EXPECT_EQ(sp(0, 3), 10);
    EXPECT_EQ(sp(1, 0), 4);
    EXPECT_EQ(sp(1, 1), 6);
    EXPECT_EQ(sp(1, 2), 8);
    EXPECT_EQ(sp(1, 3), 10);
    EXPECT_EQ(sp(2, 0), 4);
    EXPECT_EQ(sp(2, 1), 6);
    EXPECT_EQ(sp(2, 2), 8);
    EXPECT_EQ(sp(2, 3), 10);
  }

  {
    /// [stl-for-each-zip-by-row]
    // Row-wise operation on two logical stores.
    stl::logical_store<int, 2> store1 = {{1, 2, 3, 4},  //
                                         {2, 3, 4, 5},
                                         {3, 4, 5, 6}};
    stl::logical_store<int, 2> store2 = {{3, 4, 5, 6},  //
                                         {2, 3, 4, 5},
                                         {1, 2, 3, 4}};

    // `a` and `b` are `mdspan` objects refering to the rows of `store1`
    // and `store2`.
    auto fn = [] LEGATE_HOST_DEVICE(auto a, auto b) {
      for (std::ptrdiff_t i = 0; i < a.extent(0); ++i) {
        a(i) += b(i);
      }
    };
    stl::for_each_zip(fn, stl::rows_of(store1), stl::rows_of(store2));

    // store1 now contains:
    // {
    //   {4, 6, 8, 10},
    //   {4, 6, 8, 10},
    //   {4, 6, 8, 10}
    // }
    /// [stl-for-each-zip-by-row]

    auto sp = stl::as_mdspan(store1);
    EXPECT_EQ(sp(0, 0), 4);
    EXPECT_EQ(sp(0, 1), 6);
    EXPECT_EQ(sp(0, 2), 8);
    EXPECT_EQ(sp(0, 3), 10);
    EXPECT_EQ(sp(1, 0), 4);
    EXPECT_EQ(sp(1, 1), 6);
    EXPECT_EQ(sp(1, 2), 8);
    EXPECT_EQ(sp(1, 3), 10);
    EXPECT_EQ(sp(2, 0), 4);
    EXPECT_EQ(sp(2, 1), 6);
    EXPECT_EQ(sp(2, 2), 8);
    EXPECT_EQ(sp(2, 3), 10);
  }
}

void for_each_doxy_snippets()
{
  {
    /// [stl-for-each-elements]
    // Element-wise addition of two logical stores.
    stl::logical_store<int, 2> store = {{1, 2, 3, 4},  //
                                        {2, 3, 4, 5},
                                        {3, 4, 5, 6}};

    // `a` refers to the elements of `store`.
    auto fn = [] LEGATE_HOST_DEVICE(int& a) { ++a; };
    stl::for_each(store, fn);

    // store1 now contains:
    // {
    //   {2, 3, 4, 5},
    //   {3, 4, 5, 6}
    //   {4, 5, 6, 7}
    // }
    /// [stl-for-each-elements]

    auto sp = stl::as_mdspan(store);
    EXPECT_EQ(sp(0, 0), 2);
    EXPECT_EQ(sp(0, 1), 3);
    EXPECT_EQ(sp(0, 2), 4);
    EXPECT_EQ(sp(0, 3), 5);
    EXPECT_EQ(sp(1, 0), 3);
    EXPECT_EQ(sp(1, 1), 4);
    EXPECT_EQ(sp(1, 2), 5);
    EXPECT_EQ(sp(1, 3), 6);
    EXPECT_EQ(sp(2, 0), 4);
    EXPECT_EQ(sp(2, 1), 5);
    EXPECT_EQ(sp(2, 2), 6);
    EXPECT_EQ(sp(2, 3), 7);
  }

  {
    /// [stl-for-each-by-row]
    // Row-wise operation on two logical stores.
    stl::logical_store<int, 2> store = {{1, 2, 3, 4},  //
                                        {2, 3, 4, 5},
                                        {3, 4, 5, 6}};

    // `a` is an `mdspan` object refering to the rows of `store`.
    auto fn = [] LEGATE_HOST_DEVICE(auto a) { a(0) = 42; };
    stl::for_each(stl::rows_of(store), fn);

    // store1 now contains:
    // {
    //   {42, 2, 3, 4},
    //   {42, 3, 4, 5},
    //   {42, 4, 5, 6}
    // }
    /// [stl-for-each-by-row]

    auto sp = stl::as_mdspan(store);
    EXPECT_EQ(sp(0, 0), 42);
    EXPECT_EQ(sp(0, 1), 2);
    EXPECT_EQ(sp(0, 2), 3);
    EXPECT_EQ(sp(0, 3), 4);
    EXPECT_EQ(sp(1, 0), 42);
    EXPECT_EQ(sp(1, 1), 3);
    EXPECT_EQ(sp(1, 2), 4);
    EXPECT_EQ(sp(1, 3), 5);
    EXPECT_EQ(sp(2, 0), 42);
    EXPECT_EQ(sp(2, 1), 4);
    EXPECT_EQ(sp(2, 2), 5);
    EXPECT_EQ(sp(2, 3), 6);
  }
}

// NOLINTEND(readability-magic-numbers, misc-const-correctness)

}  // namespace

TEST_F(STL, ForEach1D) { for_each_1D(); }

TEST_F(STL, ForEachZipDoxySnippets) { for_each_zip_doxy_snippets(); }

TEST_F(STL, ForEachDoxySnippets) { for_each_doxy_snippets(); }
