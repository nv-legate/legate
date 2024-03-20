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

void TestTransformSingleInPlace()
{
  constexpr std::size_t extents[] = {4, 5};
  auto store                      = stl::create_store<std::int64_t>(extents);

  // Stateless extended lambdas work with both clang CUDA and nvcc
  auto inc = [] LEGATE_HOST_DEVICE(std::int64_t v) -> std::int64_t { return v + 2; };

  stl::fill(store, 1);
  stl::transform(store, store, inc);

  // Check the result
  for (auto i : stl::elements_of(store)) {
    EXPECT_EQ(i, 3);
  }
}

void TestTransformSingleCopy()
{
  constexpr std::size_t extents[] = {4, 5};
  auto in_store                   = stl::create_store<std::int64_t>(extents);
  auto out_store                  = stl::create_store<std::int64_t>(extents);

  // Stateless extended lambdas work with both clang CUDA and nvcc
  auto inc = [] LEGATE_HOST_DEVICE(std::int64_t v) -> std::int64_t { return v + 2; };

  stl::fill(stl::elements_of(in_store), 1);
  stl::transform(stl::elements_of(in_store), stl::elements_of(out_store), inc);

  // Check the result
  for (auto i : stl::elements_of(in_store)) {
    EXPECT_EQ(i, 1);
  }

  // Check the result
  for (auto i : stl::elements_of(out_store)) {
    EXPECT_EQ(i, 3);
  }
}

void TestTransformDoubleInPlace()
{
  constexpr std::size_t extents[] = {4, 5};
  auto store1                     = stl::create_store<std::int64_t>(extents);
  auto store2                     = stl::create_store<std::int64_t>(extents);

  // Stateless extended lambdas work with both clang CUDA and nvcc
  auto shift = [] LEGATE_HOST_DEVICE(std::int64_t a, std::int64_t b) -> std::int64_t {
    return a << b;
  };

  stl::fill(store1, 2);
  stl::fill(store2, 4);
  stl::transform(store1, store2, store1, shift);

  // Check the result
  for (auto i : stl::elements_of(store1)) {
    EXPECT_EQ(i, 32);
  }

  for (auto i : stl::elements_of(store2)) {
    EXPECT_EQ(i, 4);
  }
}

void TestTransformDoubleCopy()
{
  constexpr std::size_t extents[] = {4, 5};
  auto store1                     = stl::create_store<std::int64_t>(extents);
  auto store2                     = stl::create_store<std::int64_t>(extents);
  auto store3                     = stl::create_store<std::int64_t>(extents);

  // Stateless extended lambdas work with both clang CUDA and nvcc
  auto shift = [] LEGATE_HOST_DEVICE(std::int64_t a, std::int64_t b) -> std::int64_t {
    return a << b;
  };

  stl::fill(store1, 2);
  stl::fill(store2, 4);
  stl::transform(store1, store2, store3, shift);

  // Check the result
  for (auto i : stl::elements_of(store1)) {
    EXPECT_EQ(i, 2);
  }

  for (auto i : stl::elements_of(store2)) {
    EXPECT_EQ(i, 4);
  }

  for (auto i : stl::elements_of(store3)) {
    EXPECT_EQ(i, 32);
  }
}

void TestTransformRows()
{
  auto input = stl::create_store<std::int64_t>({3, 4});

  auto input_view = stl::as_mdspan(input);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      input_view(i, j) = i * 4 + j;
    }
  }

  // Transform by rows
  auto result = stl::create_store({3, 4}, std::int64_t{0});
  stl::transform(stl::rows_of(input),  //
                 stl::rows_of(result),
                 stl::elementwise(square()));

  auto result_view = stl::as_mdspan(result);
  EXPECT_EQ(result_view.rank(), 2);
  EXPECT_EQ(result_view.extent(0), 3);
  EXPECT_EQ(result_view.extent(1), 4);
  EXPECT_EQ(result_view(0, 0), 0);
  EXPECT_EQ(result_view(0, 1), 1);
  EXPECT_EQ(result_view(0, 2), 4);
  EXPECT_EQ(result_view(0, 3), 9);
  EXPECT_EQ(result_view(1, 0), 16);
  EXPECT_EQ(result_view(1, 1), 25);
  EXPECT_EQ(result_view(1, 2), 36);
  EXPECT_EQ(result_view(1, 3), 49);
  EXPECT_EQ(result_view(2, 0), 64);
  EXPECT_EQ(result_view(2, 1), 81);
  EXPECT_EQ(result_view(2, 2), 100);
  EXPECT_EQ(result_view(2, 3), 121);
}
}  // namespace

TEST_F(STL, TestTransformSingleInPlace) { TestTransformSingleInPlace(); }

TEST_F(STL, TestTransformSingleCopy) { TestTransformSingleCopy(); }

TEST_F(STL, TestTransformDoubleInPlace) { TestTransformDoubleInPlace(); }

TEST_F(STL, TestTransformDoubleCopy) { TestTransformDoubleCopy(); }

TEST_F(STL, TestTransformRows) { TestTransformRows(); }
