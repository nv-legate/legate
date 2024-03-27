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

#include <functional>
#include <gtest/gtest.h>
#include <numeric>

using STL = LegateSTLFixture;

namespace stl = legate::stl;

namespace {

struct square {
  template <class T>
  LEGATE_STL_ATTRIBUTE((host, device))
  T operator()(T x) const
  {
    return x * x;
  }
};

void TestTransformReduce1D()
{
  stl::logical_store<std::int64_t, 1> store{{5}};

  // fill the store with data
  auto elems = stl::elements_of(store);
  std::iota(elems.begin(), elems.end(), std::int64_t(1));

  // sum the squared elements
  auto result = stl::transform_reduce(store,  //
                                      stl::scalar(std::int64_t(0)),
                                      std::plus(),
                                      square());

  auto result_span = stl::as_mdspan(result);
  auto&& value     = result_span();
  static_assert(std::is_same_v<decltype(value), const std::int64_t&>);
  EXPECT_EQ(55, value);
}

void TestTransformReduce2D()
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
    auto init   = stl::create_store({4}, std::int64_t(0));
    auto result = stl::transform_reduce(stl::rows_of(store),  //
                                        init,
                                        stl::elementwise(std::plus()),
                                        stl::elementwise(square()));

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
    auto init        = stl::create_store({3}, std::int64_t(0));
    auto result      = stl::reduce(stl::columns_of(store), init, stl::elementwise(std::plus<>()));
    auto result_span = stl::as_mdspan(result);
    EXPECT_EQ(result_span.rank(), 1);
    EXPECT_EQ(result_span.extent(0), 3);
    EXPECT_EQ(result_span(0), 0);
    EXPECT_EQ(result_span(1), 4);
    EXPECT_EQ(result_span(2), 8);
  }
}

}  // namespace

TEST_F(STL, TestTransformReduce1D) { TestTransformReduce1D(); }

TEST_F(STL, TestTransformReduce2D) { TestTransformReduce2D(); }
