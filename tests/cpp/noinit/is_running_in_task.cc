/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <thread>

namespace test_is_running_in_task_noinit {

using IsRunningInTaskNoRuntime = ::testing::Test;

TEST_F(IsRunningInTaskNoRuntime, BeforeInit) { EXPECT_FALSE(legate::is_running_in_task()); }

TEST_F(IsRunningInTaskNoRuntime, UserThread)
{
  std::thread thd{[] { EXPECT_FALSE(legate::is_running_in_task()); }};
  thd.join();
}

}  // namespace test_is_running_in_task_noinit
