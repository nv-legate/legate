/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace transform_stack_test {

namespace {

using TransformStackUnit = DefaultFixture;

using TransformStackUnitDeathTest = TransformStackUnit;

}  // namespace

TEST_F(TransformStackUnit, ConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  auto expected  = legate::tuple<std::uint64_t>{1};
  auto color     = transform->convert_color(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformStackUnit, InvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  auto expected  = legate::tuple<std::uint64_t>{1, 2, 3};
  auto color     = transform->invert_color(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformStackUnit, Pop)
{
  auto parent = legate::make_internal_shared<legate::detail::TransformStack>();
  auto child  = std::make_unique<legate::detail::Promote>(1, 2);
  auto transform =
    legate::make_internal_shared<legate::detail::TransformStack>(std::move(child), parent);
  auto result = transform->pop();

  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->is_convertible());
  ASSERT_EQ(result->target_ndim(1), 0);
  ASSERT_TRUE(transform->identity());
  ASSERT_TRUE(transform->is_convertible());
}

TEST_F(TransformStackUnitDeathTest, DoublePop)
{
  auto parent = legate::make_internal_shared<legate::detail::TransformStack>();
  auto child  = std::make_unique<legate::detail::Promote>(1, 2);
  auto transform =
    legate::make_internal_shared<legate::detail::TransformStack>(std::move(child), parent);

  ASSERT_NO_THROW(static_cast<void>(transform->pop()));

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_EXIT(static_cast<void>(transform->pop()),
                ::testing::KilledBySignal(SIGABRT),
                "transform_ != nullptr");
  }
}

TEST_F(TransformStackUnitDeathTest, NegativePop)
{
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_EXIT(static_cast<void>(transform->pop()),
                ::testing::KilledBySignal(SIGABRT),
                "transform_ != nullptr");
  }
}

}  // namespace transform_stack_test
