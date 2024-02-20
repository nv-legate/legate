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

namespace legate_scope_guard_test {

struct ScopeGuardUnit : ::testing::Test {};

TEST_F(ScopeGuardUnit, Construct)
{
  struct Callable {
    void operator()() noexcept {}
  };

  legate::ScopeGuard<Callable> guard{Callable{}};

  EXPECT_TRUE(guard.enabled());
}

TEST_F(ScopeGuardUnit, EnableConstruct)
{
  struct Callable {
    void operator()() noexcept {}
  };

  legate::ScopeGuard<Callable> guard{Callable{}, true};

  EXPECT_TRUE(guard.enabled());

  legate::ScopeGuard<Callable> guard2{Callable{}, false};

  EXPECT_FALSE(guard2.enabled());
}

TEST_F(ScopeGuardUnit, ConstructFromHelper)
{
  auto guard = legate::make_scope_guard([]() noexcept {});

  EXPECT_TRUE(guard.enabled());
}

TEST_F(ScopeGuardUnit, ConstructAndExecute)
{
  bool executed = false;
  {
    auto guard = legate::make_scope_guard([&]() noexcept { executed = true; });

    // ensure that func was never run
    EXPECT_FALSE(executed);
    EXPECT_TRUE(guard.enabled());
    EXPECT_FALSE(executed);
  }
  EXPECT_TRUE(executed);
}

TEST_F(ScopeGuardUnit, ConstructAndExecuteNested)
{
  int executed = 0;
  {
    auto guard1 = legate::make_scope_guard([&]() noexcept { ++executed; });

    EXPECT_EQ(executed, 0);
    {
      auto guard2 = legate::make_scope_guard([&]() noexcept { ++executed; });

      EXPECT_EQ(executed, 0);
    }
    EXPECT_EQ(executed, 1);
  }
  EXPECT_EQ(executed, 2);
}

TEST_F(ScopeGuardUnit, Disable)
{
  bool executed = false;
  {
    auto guard = legate::make_scope_guard([&]() noexcept { executed = true; });

    EXPECT_TRUE(guard.enabled());
    guard.disable();
    // ensure this doesn't run the function
    EXPECT_FALSE(executed);
    EXPECT_FALSE(guard.enabled());
    // double disable should have no effect
    guard.disable();
    EXPECT_FALSE(guard.enabled());
  }
  EXPECT_FALSE(executed);
}

TEST_F(ScopeGuardUnit, Enable)
{
  bool executed = false;
  {
    auto guard = legate::make_scope_guard([&]() noexcept { executed = true; });

    EXPECT_TRUE(guard.enabled());
    guard.enable();
    // ensure this doesn't run the function
    EXPECT_FALSE(executed);
    EXPECT_TRUE(guard.enabled());
    // double enable should have no effect
    guard.enable();
    EXPECT_TRUE(guard.enabled());
  }
  EXPECT_TRUE(executed);
}

TEST_F(ScopeGuardUnit, EnableDisable)
{
  bool executed = false;
  {
    auto guard = legate::make_scope_guard([&]() noexcept { executed = true; });

    EXPECT_TRUE(guard.enabled());
    guard.disable();
    EXPECT_FALSE(guard.enabled());
    guard.enable();
    EXPECT_TRUE(guard.enabled());
  }
  EXPECT_TRUE(executed);
}

TEST_F(ScopeGuardUnit, FromMacro)
{
  bool executed1 = false;
  bool executed2 = false;
  bool executed3 = false;
  bool executed4 = false;
  {
    // clang-format off
    LEGATE_SCOPE_GUARD(
      for (auto& exec : {&executed1, &executed2, &executed3, &executed4}) {
        *exec = true;
      }
    );
    // clang-format on
  }
  EXPECT_TRUE(executed1);
  EXPECT_TRUE(executed2);
  EXPECT_TRUE(executed3);
  EXPECT_TRUE(executed4);
}

}  // namespace legate_scope_guard_test
