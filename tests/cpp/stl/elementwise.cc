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

namespace {

struct square {
  template <class T>
  LEGATE_HOST_DEVICE T operator()(T x) const
  {
    return x * x;
  }
};

void TestElementwiseRowOperation()
{
  auto input  = stl::create_store<std::int64_t>({3, 4});
  auto result = stl::create_store({3, 4}, std::int64_t{0});

  // Fill the input
  {
    auto input_view = stl::as_mdspan(input);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        input_view(i, j) = i * 4 + j;
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
    auto res_row            = *res_iter;
    stl::as_mdspan(res_row) = stl::elementwise(square())(stl::as_mdspan(*in_iter));
  }

  EXPECT_EQ(result_view(0, 0), 0);
  EXPECT_EQ(result_view(0, 1), 1);
  EXPECT_EQ(result_view(0, 2), 4);
  EXPECT_EQ(result_view(0, 3), 9);

  {
    ++in_iter;
    ++res_iter;
    auto res_row            = *res_iter;
    stl::as_mdspan(res_row) = stl::elementwise(square())(stl::as_mdspan(*in_iter));
  }

  EXPECT_EQ(result_view(1, 0), 16);
  EXPECT_EQ(result_view(1, 1), 25);
  EXPECT_EQ(result_view(1, 2), 36);
  EXPECT_EQ(result_view(1, 3), 49);

  {
    ++in_iter;
    ++res_iter;
    auto res_row            = *res_iter;
    stl::as_mdspan(res_row) = stl::elementwise(square())(stl::as_mdspan(*in_iter));
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
}  // namespace

TEST_F(STL, TestElementwiseRowOperation) { TestElementwiseRowOperation(); }
