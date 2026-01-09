/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_slice_unit {

namespace {

using LogicalStoreSliceUnit = DefaultFixture;

constexpr std::int32_t SCALAR_VALUE = 10;

class SliceStoreTest
  : public LogicalStoreSliceUnit,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t, legate::Slice, std::vector<std::uint64_t>, bool, bool>> {};

class SliceBoundStoreTest : public SliceStoreTest {};

class SliceScalarStoreTest : public SliceStoreTest {};

class NegativeSliceStoreDimTest : public LogicalStoreSliceUnit,
                                  public ::testing::WithParamInterface<std::int32_t> {};

class NegativeSliceBoundStoreSliceTest
  : public LogicalStoreSliceUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, std::int32_t, legate::Slice>> {};

class NegativeSliceScalarStoreSliceTest : public LogicalStoreSliceUnit,
                                          public ::testing::WithParamInterface<legate::Slice> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreSliceUnit,
  SliceBoundStoreTest,
  ::testing::Values(
    std::make_tuple(1, legate::Slice{-2, -1}, std::vector<std::uint64_t>{4, 1}, true, true),
    std::make_tuple(1, legate::Slice{1, 2}, std::vector<std::uint64_t>{4, 1}, true, true),
    std::make_tuple(0, legate::Slice{}, std::vector<std::uint64_t>{4, 3}, false, true),
    std::make_tuple(0, legate::Slice{0, 0}, std::vector<std::uint64_t>{0, 3}, false, false),
    std::make_tuple(1, legate::Slice{1, 1}, std::vector<std::uint64_t>{4, 0}, false, false),
    std::make_tuple(0, legate::Slice{-9, -8}, std::vector<std::uint64_t>{0, 3}, false, false),
    std::make_tuple(0, legate::Slice{-8, -10}, std::vector<std::uint64_t>{0, 3}, false, false),
    std::make_tuple(0, legate::Slice{1, 1}, std::vector<std::uint64_t>{0, 3}, false, false),
    std::make_tuple(0, legate::Slice{-1, 0}, std::vector<std::uint64_t>{0, 3}, false, false),
    std::make_tuple(0, legate::Slice{-1, 1}, std::vector<std::uint64_t>{0, 3}, false, false),
    std::make_tuple(0, legate::Slice{10, 8}, std::vector<std::uint64_t>{0, 3}, false, false)));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreSliceUnit,
  SliceScalarStoreTest,
  ::testing::Values(
    std::make_tuple(0, legate::Slice{-2, -1}, std::vector<std::uint64_t>{0}, false, false),
    std::make_tuple(0, legate::Slice{}, std::vector<std::uint64_t>{1}, false, true),
    std::make_tuple(0, legate::Slice{0, 0}, std::vector<std::uint64_t>{0}, false, false),
    std::make_tuple(0, legate::Slice{1, 1}, std::vector<std::uint64_t>{0}, false, false)));

INSTANTIATE_TEST_SUITE_P(LogicalStoreSliceUnit,
                         NegativeSliceStoreDimTest,
                         ::testing::Values(-1, LEGATE_MAX_DIM));

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreSliceUnit,
  NegativeSliceBoundStoreSliceTest,
  ::testing::Values(std::make_tuple(legate::Shape{4, 3}, 0, legate::Slice{4, 5}),
                    std::make_tuple(legate::Shape{4, 3}, 1, legate::Slice{1, 4})));

INSTANTIATE_TEST_SUITE_P(LogicalStoreSliceUnit,
                         NegativeSliceScalarStoreSliceTest,
                         ::testing::Values(legate::Slice{0, 2},
                                           legate::Slice{-1, 2},
                                           legate::Slice{2, 3}));

}  // namespace

TEST_P(SliceBoundStoreTest, Basic)
{
  const auto [dim, slice_range, slice_shape, transformed, overlaps] = GetParam();
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 3}, legate::int64());
  auto slice   = store.slice(dim, slice_range);

  ASSERT_EQ(slice.extents().data(), slice_shape);
  ASSERT_EQ(slice.transformed(), transformed);
  ASSERT_EQ(slice.type(), store.type());
  ASSERT_EQ(slice.overlaps(store), overlaps);
  ASSERT_EQ(slice.dim(), store.dim());
}

TEST_P(SliceScalarStoreTest, Basic)
{
  const auto [dim, slice_range, slice_shape, transformed, overlaps] = GetParam();
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto slice   = store.slice(dim, slice_range);

  ASSERT_EQ(slice.extents().data(), slice_shape);
  ASSERT_EQ(slice.transformed(), transformed);
  ASSERT_EQ(slice.type(), store.type());
  ASSERT_EQ(slice.overlaps(store), overlaps);
  ASSERT_EQ(slice.dim(), store.dim());
}

TEST_P(NegativeSliceStoreDimTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4}, legate::int64());

  ASSERT_THROW(static_cast<void>(store.slice(GetParam(), legate::Slice{})), std::invalid_argument);
}

TEST_P(NegativeSliceBoundStoreSliceTest, Basic)
{
  const auto [shape, dim, slice_range] = GetParam();
  auto runtime                         = legate::Runtime::get_runtime();
  auto store                           = runtime->create_store(shape, legate::int64());

  ASSERT_THROW(static_cast<void>(store.slice(dim, slice_range)), std::invalid_argument);
}

TEST_P(NegativeSliceStoreDimTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.slice(GetParam(), legate::Slice{})), std::invalid_argument);
}

TEST_P(NegativeSliceScalarStoreSliceTest, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.slice(0, GetParam())), std::invalid_argument);
}

}  // namespace logical_store_slice_unit
