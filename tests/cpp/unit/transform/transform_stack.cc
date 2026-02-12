/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/transform_stack.h>

#include <legate/data/detail/transform/promote.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gmock/gmock.h>
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
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1};
  auto color     = transform->convert_color(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformStackUnit, ConvertColorNonIdentity)
{
  // Create parent (identity)
  auto parent = legate::make_internal_shared<legate::detail::TransformStack>();
  // Create child transform (Promote adds a dimension)
  auto child = std::make_unique<legate::detail::Promote>(2, 3);
  // Create non-identity TransformStack
  auto transform =
    legate::make_internal_shared<legate::detail::TransformStack>(std::move(child), parent);

  // Input: {1, 2}, after Promote at dim 2: {1, 2, 0}
  auto input  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2};
  auto result = transform->convert_color(input);

  ASSERT_THAT(result, ::testing::ElementsAre(1, 2, 0));
}

TEST_F(TransformStackUnit, ConvertColorNestedNonIdentity)
{
  // Create grandparent (identity)
  auto grandparent = legate::make_internal_shared<legate::detail::TransformStack>();
  // Create parent transform (Promote at dim 1)
  auto parent_child = std::make_unique<legate::detail::Promote>(1, 2);
  auto parent       = legate::make_internal_shared<legate::detail::TransformStack>(
    std::move(parent_child), grandparent);
  // Create child transform (Promote at dim 2, must be <= size after parent)
  auto child = std::make_unique<legate::detail::Promote>(2, 3);
  // Create nested non-identity TransformStack
  auto transform =
    legate::make_internal_shared<legate::detail::TransformStack>(std::move(child), parent);

  // Input: {1}, after parent Promote at dim 1: {1, 0}, after child Promote at dim 2: {1, 0, 0}
  auto input  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1};
  auto result = transform->convert_color(input);

  ASSERT_THAT(result, ::testing::ElementsAre(1, 0, 0));
}

TEST_F(TransformStackUnit, InvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2, 3};
  auto color     = transform->invert_color(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformStackUnit, InvertColorNonIdentity)
{
  // Create parent (identity)
  auto parent = legate::make_internal_shared<legate::detail::TransformStack>();
  // Create child transform (Promote adds a dimension at index 2)
  auto child = std::make_unique<legate::detail::Promote>(2, 3);
  // Create non-identity TransformStack
  auto transform =
    legate::make_internal_shared<legate::detail::TransformStack>(std::move(child), parent);

  // Input: {1, 2, 0}, after Promote invert (remove dim 2): {1, 2}
  auto input  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2, 0};
  auto result = transform->invert_color(input);

  ASSERT_THAT(result, ::testing::ElementsAre(1, 2));
}

TEST_F(TransformStackUnit, InvertColorNestedNonIdentity)
{
  // Create grandparent (identity)
  auto grandparent = legate::make_internal_shared<legate::detail::TransformStack>();
  // Create parent transform (Promote at dim 1)
  auto parent_child = std::make_unique<legate::detail::Promote>(1, 2);
  auto parent       = legate::make_internal_shared<legate::detail::TransformStack>(
    std::move(parent_child), grandparent);
  // Create child transform (Promote at dim 2)
  auto child = std::make_unique<legate::detail::Promote>(2, 3);
  // Create nested non-identity TransformStack
  auto transform =
    legate::make_internal_shared<legate::detail::TransformStack>(std::move(child), parent);

  // Input: {1, 0, 0}, invert child (remove dim 2): {1, 0}, invert parent (remove dim 1): {1}
  auto input  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 0, 0};
  auto result = transform->invert_color(input);

  ASSERT_THAT(result, ::testing::ElementsAre(1));
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
                ::testing::KilledBySignal{SIGABRT},
                "transform_ != nullptr");
  }
}

TEST_F(TransformStackUnitDeathTest, NegativePop)
{
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_EXIT(static_cast<void>(transform->pop()),
                ::testing::KilledBySignal{SIGABRT},
                "transform_ != nullptr");
  }
}

}  // namespace transform_stack_test
