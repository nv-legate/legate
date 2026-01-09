/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/stl.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <numeric>
#include <utilities/utilities.h>

using STL = DefaultFixture;

namespace stl = legate::experimental::stl;

namespace {

// NOLINTBEGIN(readability-magic-numbers)

class Square {
 public:
  template <class T>
  LEGATE_HOST_DEVICE T operator()(T x) const
  {
    return x * x;
  }
};

// clang-tidy complains that "1D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void test_transform_reduce_1D()
{
  stl::logical_store<std::int64_t, 1> store{{5}};

  // fill the store with data
  auto elems = stl::elements_of(store);
  std::iota(elems.begin(), elems.end(), std::int64_t{1});

  // sum the squared elements
  auto result = stl::transform_reduce(store,  //
                                      stl::scalar(std::int64_t{0}),
                                      std::plus<>{},
                                      Square{});

  auto result_span = stl::as_mdspan(result);
  auto&& value     = result_span();
  static_assert(std::is_same_v<decltype(value), const std::int64_t&>);
  EXPECT_EQ(55, value);
}

// clang-tidy complains that "1D" is not lower case
// NOLINTNEXTLINE(readability-identifier-naming)
void test_transform_reduce_2D()
{
  auto store = stl::create_store<std::int64_t>({3, 4});

  auto store_span = stl::as_mdspan(store);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      store_span(i, j) = i;
    }
  }

  // Reduce by rows
  {
    auto init   = stl::create_store({4}, std::int64_t{0});
    auto result = stl::transform_reduce(stl::rows_of(store),  //
                                        init,
                                        stl::elementwise(std::plus<>{}),
                                        stl::elementwise(Square{}));

    auto result_span = stl::as_mdspan(result);
    EXPECT_EQ(result_span.rank(), 1);
    EXPECT_EQ(result_span.extent(0), 4);
    EXPECT_EQ(result_span(0), 5);
    EXPECT_EQ(result_span(1), 5);
    EXPECT_EQ(result_span(2), 5);
    EXPECT_EQ(result_span(3), 5);
  }

  // Reduce by columns
  {
    auto init        = stl::create_store({3}, std::int64_t{0});
    auto result      = stl::reduce(stl::columns_of(store), init, stl::elementwise(std::plus<>{}));
    auto result_span = stl::as_mdspan(result);
    EXPECT_EQ(result_span.rank(), 1);
    EXPECT_EQ(result_span.extent(0), 3);
    EXPECT_EQ(result_span(0), 0);
    EXPECT_EQ(result_span(1), 4);
    EXPECT_EQ(result_span(2), 8);
  }
}

void transform_reduce_doxy_snippets()
{
  /// [1D unary transform_reduce]
  stl::logical_store<std::int64_t, 1> store{{5}};

  // fill the store with data. The store will contain {1, 2, 3, 4, 5}
  auto elems = stl::elements_of(store);
  std::iota(elems.begin(), elems.end(), std::int64_t{1});

  // a host/device lambda to square the elements
  auto square = [] LEGATE_HOST_DEVICE(std::int64_t x) { return x * x; };

  // sum the squared elements
  auto result = stl::transform_reduce(store, stl::scalar<std::int64_t>(0), std::plus<>{}, square);

  auto result_span  = stl::as_mdspan(result);
  auto result_value = result_span();  // index into the 0-D mdspan
  // result_value is 55
  /// [1D unary transform_reduce]
  EXPECT_EQ(55, result_value);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace

TEST_F(STL, TestTransformReduce1D) { test_transform_reduce_1D(); }

TEST_F(STL, TestTransformReduce2D) { test_transform_reduce_2D(); }

TEST_F(STL, TransformReduceDoxySnippets) { transform_reduce_doxy_snippets(); }
