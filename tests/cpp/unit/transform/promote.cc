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

namespace transform_promote_test {

namespace {

using TransformPromoteUnit = DefaultFixture;

}  // namespace

TEST_F(TransformPromoteUnit, PromoteConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Promote>(2, 3);
  auto color     = transform->convert_color(legate::tuple<std::uint64_t>{1, 3});
  auto expected  = legate::tuple<std::uint64_t>{1, 3, 0};

  ASSERT_EQ(color, expected);
  ASSERT_EQ(transform->target_ndim(1), 0);
  ASSERT_TRUE(transform->is_convertible());
}

TEST_F(TransformPromoteUnit, PromoteConvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Promote>(2, 3);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] { static_cast<void>(transform->convert_color(legate::tuple<std::uint64_t>{1})); },
      ::testing::ThrowsMessage<std::out_of_range>(
        ::testing::HasSubstr("Index 2 out of range [0, 2)")));
  }
}

TEST_F(TransformPromoteUnit, PromoteInvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Promote>(2, 3);
  auto color     = transform->invert_color(legate::tuple<std::uint64_t>{1, 2, 3});
  auto expected  = legate::tuple<std::uint64_t>{1, 2};

  ASSERT_EQ(color, expected);
}

TEST_F(TransformPromoteUnit, PromoteInvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Promote>(2, 3);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] { static_cast<void>(transform->invert_color(legate::tuple<std::uint64_t>{1, 2})); },
      ::testing::ThrowsMessage<std::out_of_range>(
        ::testing::HasSubstr("Index 2 out of range [0, 2)")));
  }
}

}  // namespace transform_promote_test
