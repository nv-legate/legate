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

#include "core/utilities/scope_guard.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace legate_scope_fail_test {

struct ScopeFailUnit : ::testing::Test {};

TEST_F(ScopeFailUnit, Construct)
{
  struct Callable {
    void operator()() noexcept {}
  };

  legate::ScopeFail<Callable> guard{Callable{}};
  // nothing to do...
}

TEST_F(ScopeFailUnit, ConstructFromHelper)
{
  auto guard = legate::make_scope_fail([]() noexcept {});
  // nothing to do
}

TEST_F(ScopeFailUnit, ConstructAndExecute)
{
  auto executed = false;
  {
    auto guard = legate::make_scope_fail([&]() noexcept { executed = true; });

    // ensure that func was never run
    EXPECT_FALSE(executed);
  }
  EXPECT_FALSE(executed);
}

TEST_F(ScopeFailUnit, ConstructAndExecuteThrow)
{
  auto executed = false;
  try {
    auto guard = legate::make_scope_fail([&]() noexcept { executed = true; });

    // ensure that func was never run
    EXPECT_FALSE(executed);
    throw std::runtime_error{"foo"};
  } catch (const std::exception& e) {
    ASSERT_STREQ(e.what(), "foo");
    EXPECT_TRUE(executed);
  }
  EXPECT_TRUE(executed);
}

TEST_F(ScopeFailUnit, ConstructAndExecuteUnrelatedThrow)
{
  auto executed         = false;
  auto caught_outer_exn = false;
  try {
    auto guard      = legate::make_scope_fail([&]() noexcept { executed = true; });
    auto thrower    = [] { throw std::runtime_error{"from inside lambda"}; };
    auto caught_exn = false;

    try {
      thrower();
    } catch (const std::exception& e) {
      caught_exn = true;
      ASSERT_STREQ(e.what(), "from inside lambda");
      EXPECT_FALSE(executed);
    }
    // To ensure the compiler doesn't optimize this stuff away...
    EXPECT_TRUE(caught_exn);
    // ensure that func was never run
    EXPECT_FALSE(executed);
    throw std::runtime_error{"foo"};
  } catch (const std::exception& e) {
    caught_outer_exn = true;
    ASSERT_STREQ(e.what(), "foo");
  }
  EXPECT_TRUE(caught_outer_exn);
  EXPECT_TRUE(executed);
}

TEST_F(ScopeFailUnit, FromMacro)
{
  auto executed1  = false;
  auto executed2  = false;
  auto executed3  = false;
  auto executed4  = false;
  auto caught_exn = false;
  try {
    // clang-format off
    LEGATE_SCOPE_FAIL(
      for (auto& exec : {&executed1, &executed2, &executed3, &executed4}) {
        *exec = true;
      }
    );
    // clang-format on

    throw std::runtime_error{"failure"};
  } catch (const std::exception& e) {
    caught_exn = true;
    ASSERT_STREQ(e.what(), "failure");
  }
  EXPECT_TRUE(caught_exn);
  EXPECT_TRUE(executed1);
  EXPECT_TRUE(executed2);
  EXPECT_TRUE(executed3);
  EXPECT_TRUE(executed4);
}

}  // namespace legate_scope_fail_test
