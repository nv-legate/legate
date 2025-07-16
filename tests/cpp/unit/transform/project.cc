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

namespace transform_project_test {

namespace {

using TransformProjectUnit = DefaultFixture;

}  // namespace

TEST_F(TransformProjectUnit, ProjectConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Project>(1, 2);
  auto color =
    transform->convert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 1});

  ASSERT_THAT(color, ::testing::ElementsAre(3));
  ASSERT_EQ(transform->target_ndim(0), 1);
  ASSERT_TRUE(transform->is_convertible());
}

TEST_F(TransformProjectUnit, ProjectConvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Project>(1, 2);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(
          transform->convert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{}));
      },
      ::testing::ThrowsMessage<std::out_of_range>(
        ::testing::HasSubstr("Index 1 out of range [0, 0)")));
  }
}

TEST_F(TransformProjectUnit, ProjectInvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Project>(2, 3);
  auto color =
    transform->invert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 3, 0};

  ASSERT_EQ(color, expected);
}

TEST_F(TransformProjectUnit, ProjectInvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Project>(2, 3);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(
          transform->invert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2}));
      },
      ::testing::ThrowsMessage<std::out_of_range>(
        ::testing::HasSubstr("Index 2 out of range [0, 2)")));
  }
}

}  // namespace transform_project_test
