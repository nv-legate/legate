/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace transform_transpose_test {

namespace {

using TransformTransposeUnit = DefaultFixture;

}  // namespace

TEST_F(TransformTransposeUnit, TransposeConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Transpose>(
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{2, 1, 0});
  auto color =
    transform->convert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 2, 1};

  ASSERT_EQ(color, expected);
  ASSERT_EQ(transform->target_ndim(2), 2);
  ASSERT_TRUE(transform->is_convertible());
}

TEST_F(TransformTransposeUnit, TransposeConvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Transpose>(
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{-1, 0});

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(transform->convert_color(
          legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 1}));
      },
      ::testing::ThrowsMessage<std::out_of_range>(
        ::testing::HasSubstr("mapping [-1, 0] contains negative elements")));
  }
}

TEST_F(TransformTransposeUnit, TransposeInvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Transpose>(
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{4, 1, 0, 3, 2});
  auto color = transform->invert_color(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{4, 1, 0, 3, 2});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 1, 2, 3, 4};

  ASSERT_EQ(color, expected);
}

TEST_F(TransformTransposeUnit, TransposeInvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Transpose>(
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{0});

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(transform->invert_color(
          legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2}));
      },
      ::testing::ThrowsMessage<std::out_of_range>(
        ::testing::HasSubstr("mapping size 1 != container size 2")));
  }
}

}  // namespace transform_transpose_test
