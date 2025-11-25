/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/operation/detail/execution_fence.h>
#include <legate/operation/detail/mapping_fence.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/streaming/analysis.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_nested_streaming_scopes {

namespace {

auto make_policy(legate::StreamingMode mode, std::uint32_t od_factor = 1)
{
  return legate::ParallelPolicy{}.with_streaming(mode).with_overdecompose_factor(od_factor);
}

void make_equal_scopes(legate::StreamingMode mode)
{
  [[maybe_unused]] auto outer = legate::Scope{make_policy(mode)};
  {
    [[maybe_unused]] auto inner = legate::Scope{make_policy(mode)};
  }
}

void make_diff_mode_scopes(legate::StreamingMode outer_mode, legate::StreamingMode inner_mode)
{
  [[maybe_unused]] auto outer = legate::Scope{make_policy(outer_mode)};
  {
    [[maybe_unused]] auto inner = legate::Scope{make_policy(inner_mode)};
  }
}

// different policy, same mode
void make_diff_od_scopes(legate::StreamingMode outer_mode,
                         std::uint32_t outer_od,
                         legate::StreamingMode inner_mode,
                         std::uint32_t inner_od)
{
  [[maybe_unused]] auto outer = legate::Scope{make_policy(outer_mode, outer_od)};
  {
    [[maybe_unused]] auto inner = legate::Scope{make_policy(inner_mode, inner_od)};
  }
}

}  // namespace

class NestedScopes : public DefaultFixture,
                     public ::testing::WithParamInterface<legate::StreamingMode> {};

INSTANTIATE_TEST_SUITE_P(NestedScopesSuite,
                         NestedScopes,
                         ::testing::ValuesIn({legate::StreamingMode::RELAXED,
                                              legate::StreamingMode::STRICT}));

TEST_P(NestedScopes, EqualScopes)
{
  const legate::StreamingMode mode = GetParam();
  ASSERT_NO_THROW(make_equal_scopes(mode));
}

TEST_P(NestedScopes, OuterOffInnerAny)
{
  const legate::StreamingMode mode = GetParam();
  ASSERT_NO_THROW(make_diff_mode_scopes(legate::StreamingMode::OFF, mode));
}

TEST_F(NestedScopes, OuterRelaxedInnerOFF)
{
  ASSERT_NO_THROW(
    make_diff_mode_scopes(legate::StreamingMode::RELAXED, legate::StreamingMode::OFF));
}

TEST_F(NestedScopes, OuterRelaxedInnerStrict)
{
  ASSERT_NO_THROW(
    make_diff_mode_scopes(legate::StreamingMode::RELAXED, legate::StreamingMode::STRICT));
}

TEST_F(NestedScopes, OuterRelaxedInnerStrictDiffOD)
{
  ASSERT_NO_THROW(
    make_diff_od_scopes(legate::StreamingMode::RELAXED, 4, legate::StreamingMode::RELAXED, 8));
}

TEST_F(NestedScopes, OuterStrictInnerOff)
{
  ASSERT_THAT(
    [&] {
      make_diff_mode_scopes(legate::StreamingMode::STRICT, legate::StreamingMode::OFF);
      return 0;
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("cannot nest a non-streaming scope inside a STRICT streaming scope")));
}

TEST_F(NestedScopes, OuterStrictInnerRelaxed)
{
  ASSERT_THAT(
    [&] {
      make_diff_mode_scopes(legate::StreamingMode::STRICT, legate::StreamingMode::RELAXED);
      return 0;
    },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
      "cannot nest a RELAXED streaming scope inside a STRICT streaming scope")));
}

TEST_F(NestedScopes, OuterStrictInnerStrictDiffOD)
{
  ASSERT_THAT(
    [&] {
      make_diff_od_scopes(legate::StreamingMode::STRICT, 4, legate::StreamingMode::STRICT, 8);
      return 0;
    },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
      "cannot change the parallel policy when nesting a scope inside a STRICT streaming scope")));
}

TEST_F(NestedScopes, OuterOffInnerOffDiffOD)
{
  ASSERT_NO_THROW(
    make_diff_od_scopes(legate::StreamingMode::OFF, 4, legate::StreamingMode::OFF, 8));
}

}  // namespace test_nested_streaming_scopes
