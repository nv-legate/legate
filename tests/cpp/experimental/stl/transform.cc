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

#include <legate/experimental/stl.hpp>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

using STL = DefaultFixture;

namespace stl = legate::experimental::stl;

// NOLINTBEGIN(readability-magic-numbers, misc-const-correctness)

namespace {

class Square {
 public:
  template <class T>
  LEGATE_HOST_DEVICE T operator()(T x) const
  {
    return x * x;
  }
};

void test_transform_single_inplace()
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

void test_transform_single_copy()
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

void test_transform_double_inplace()
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

void test_transform_double_copy()
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

void test_transform_rows()
{
  auto input = stl::create_store<std::int64_t>({3, 4});

  auto input_view = stl::as_mdspan(input);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      input_view(i, j) = static_cast<std::int64_t>(i) * 4 + j;
    }
  }

  // Transform by rows
  auto result = stl::create_store({3, 4}, std::int64_t{0});
  stl::transform(stl::rows_of(input),  //
                 stl::rows_of(result),
                 stl::elementwise(Square{}));

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

void transform_doxy_snippets()
{
  {                                                             /// [stl-unary-transform-2d]
    stl::logical_store<std::int64_t, 2> input = {{0, 1, 2, 3},  //
                                                 {4, 5, 6, 7},
                                                 {8, 9, 10, 11}};

    // Transform by rows
    auto result = stl::create_store({3, 4}, std::int64_t{0});
    stl::transform(stl::rows_of(input),  //
                   stl::rows_of(result),
                   stl::elementwise(Square()));

    // `result` now contains the squares of the elements:
    //     [[0   1   4   9]
    //      [16 25  36  49]
    //      [64 81 100 121]]
    /// [stl-unary-transform-2d]

    auto sp = stl::as_mdspan(result);
    EXPECT_EQ(sp(0, 0), 0);
    EXPECT_EQ(sp(0, 1), 1);
    EXPECT_EQ(sp(0, 2), 4);
    EXPECT_EQ(sp(0, 3), 9);
    EXPECT_EQ(sp(1, 0), 16);
    EXPECT_EQ(sp(1, 1), 25);
    EXPECT_EQ(sp(1, 2), 36);
    EXPECT_EQ(sp(1, 3), 49);
    EXPECT_EQ(sp(2, 0), 64);
    EXPECT_EQ(sp(2, 1), 81);
    EXPECT_EQ(sp(2, 2), 100);
    EXPECT_EQ(sp(2, 3), 121);
  }

  {
    /// [stl-binary-transform-2d]
    std::size_t extents[] = {4, 5};
    auto store1           = stl::create_store<std::int64_t>(extents);
    auto store2           = stl::create_store<std::int64_t>(extents);
    auto store3           = stl::create_store<std::int64_t>(extents);

    // Stateless extended lambdas work with both clang CUDA and nvcc
    auto shift = [] LEGATE_HOST_DEVICE(std::int64_t a, std::int64_t b) {  //
      return a << b;
    };

    stl::fill(store1, 2);
    stl::fill(store2, 4);
    stl::transform(store1, store2, store3, shift);

    // `store3` now contains the elements:
    //     [[32 32 32 32 32]
    //      [32 32 32 32 32]
    //      [32 32 32 32 32]
    //      [32 32 32 32 32]]
    /// [stl-binary-transform-2d]

    for (auto i : stl::elements_of(store3)) {
      EXPECT_EQ(i, 32);
    }
  }
}

// NOLINTEND(readability-magic-numbers, misc-const-correctness)

}  // namespace

TEST_F(STL, TestTransformSingleInPlace) { test_transform_single_inplace(); }

TEST_F(STL, TestTransformSingleCopy) { test_transform_single_copy(); }

TEST_F(STL, TestTransformDoubleInPlace) { test_transform_double_inplace(); }

TEST_F(STL, TestTransformDoubleCopy) { test_transform_double_copy(); }

TEST_F(STL, TestTransformRows) { test_transform_rows(); }

TEST_F(STL, TransformDoxySnippets) { transform_doxy_snippets(); }
