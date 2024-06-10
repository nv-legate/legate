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

#include "core/experimental/stl.hpp"
#include "utilities/utilities.h"

#include <functional>
#include <gtest/gtest.h>
#include <numeric>

using STL = LegateSTLFixture;

namespace stl = legate::experimental::stl;

namespace {

// NOLINTBEGIN(readability-magic-numbers, misc-const-correctness)

// clang-tidy complains that "1D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void test_reduce_1D()
{
  auto store = stl::create_store({5}, std::int64_t{1});
  auto init  = stl::create_store({}, std::int64_t{1});

  // fill the store with data
  auto elems = stl::elements_of(store);
  std::iota(elems.begin(), elems.end(), std::int64_t{1});

  auto result = stl::reduce(store, init, std::plus<>());

  auto result_span = stl::as_mdspan(result);
  auto&& value     = result_span();
  static_assert(std::is_same_v<decltype(value), const std::int64_t&>);
  EXPECT_EQ(16, value);
}

// clang-tidy complains that "2D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void test_reduce_2D()
{
  auto store = stl::create_store({3, 4}, std::int64_t{1});

  auto store_span = stl::as_mdspan(store);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      store_span(i, j) = i;
    }
  }

  // Reduce by rows
  {
    auto init        = stl::create_store({4}, std::int64_t{0});
    auto result      = stl::reduce(stl::rows_of(store), init, stl::elementwise(std::plus<>()));
    auto result_span = stl::as_mdspan(result);
    EXPECT_EQ(result_span.rank(), 1);
    EXPECT_EQ(result_span.extent(0), 4);
    EXPECT_EQ(result_span(0), 3);
    EXPECT_EQ(result_span(1), 3);
    EXPECT_EQ(result_span(2), 3);
    EXPECT_EQ(result_span(3), 3);
  }

  // Reduce by columns
  {
    auto init        = stl::create_store({3}, std::int64_t{0});
    auto result      = stl::reduce(stl::columns_of(store), init, stl::elementwise(std::plus<>()));
    auto result_span = stl::as_mdspan(result);
    EXPECT_EQ(result_span.rank(), 1);
    EXPECT_EQ(result_span.extent(0), 3);
    EXPECT_EQ(result_span(0), 0);
    EXPECT_EQ(result_span(1), 4);
    EXPECT_EQ(result_span(2), 8);
  }
}

void reduce_doxy_snippets()
{
  {
    /// [stl-reduce-1d]
    auto store = stl::create_store({5}, std::int64_t{1});
    auto init  = stl::create_store({}, std::int64_t{1});

    // fill the store with data
    auto elems = stl::elements_of(store);
    std::iota(elems.begin(), elems.end(), std::int64_t{1});

    auto result = stl::reduce(store, init, std::plus<>());

    auto result_span = stl::as_mdspan(result);
    auto&& value     = result_span();
    static_assert(std::is_same_v<decltype(value), const std::int64_t&>);
    EXPECT_EQ(16, value);
    /// [stl-reduce-1d]
  }

  {
    /// [stl-reduce-2d]
    stl::logical_store<std::int64_t, 2> store = {{0, 0, 0, 0},  //
                                                 {1, 1, 1, 1},
                                                 {2, 2, 2, 2}};

    // Reduce by rows
    {
      auto init        = stl::create_store({4}, std::int64_t{0});
      auto result      = stl::reduce(stl::rows_of(store), init, stl::elementwise(std::plus<>()));
      auto result_span = stl::as_mdspan(result);
      EXPECT_EQ(result_span.rank(), 1);
      EXPECT_EQ(result_span.extent(0), 4);
      EXPECT_EQ(result_span(0), 3);
      EXPECT_EQ(result_span(1), 3);
      EXPECT_EQ(result_span(2), 3);
      EXPECT_EQ(result_span(3), 3);
    }

    // Reduce by columns
    {
      auto init        = stl::create_store({3}, std::int64_t{0});
      auto result      = stl::reduce(stl::columns_of(store), init, stl::elementwise(std::plus<>()));
      auto result_span = stl::as_mdspan(result);
      EXPECT_EQ(result_span.rank(), 1);
      EXPECT_EQ(result_span.extent(0), 3);
      EXPECT_EQ(result_span(0), 0);
      EXPECT_EQ(result_span(1), 4);
      EXPECT_EQ(result_span(2), 8);
    }
    /// [stl-reduce-2d]
  }
}

// NOLINTEND(readability-magic-numbers, misc-const-correctness)

}  // namespace

TEST_F(STL, TestReduce1D) { test_reduce_1D(); }

TEST_F(STL, TestReduce2D) { test_reduce_2D(); }

TEST_F(STL, ReduceDoxySnippets) { reduce_doxy_snippets(); }

// TODO(eniebler): Add back support for `make_reduction` once
// the following test passes.
//
// TEST_F(STL, MakeReductionDoxySnippet) {
//   /// [make-reduction-doxy-snippet]
//   // Build a legate reduction object for subtracting integers.
//   auto diff = stl::make_reduction<int, 0>(std::minus<>{}, std::plus<>{});

//   // Apply the reduction to a 1-D logical store.
//   stl::logical_store<int, 1> store = {std::in_place, {1, 2, 3, 4, 5}};
//   auto result = stl::reduce(store, stl::scalar(42), diff);

//   // The result is 42 - 1 - 2 - 3 - 4 - 5 = 27.
//   EXPECT_EQ(stl::as_mdspan(result)(), 27);
//   /// [make-reduction-doxy-snippet]
// }
