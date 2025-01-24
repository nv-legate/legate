/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <thread>
#include <utilities/utilities.h>

namespace test_is_running_in_task {

struct Checker : public legate::LegateTask<Checker> {
  static constexpr auto TASK_ID = legate::LocalTaskID{0};
  static void cpu_variant(legate::TaskContext /*context*/)
  {
    EXPECT_TRUE(legate::is_running_in_task());
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_is_running_in_task";
  static void registration_callback(legate::Library library)
  {
    Checker::register_variants(library);
  }
};

class IsRunningInTask : public RegisterOnceFixture<Config> {};

TEST_F(IsRunningInTask, Toplevel) { EXPECT_FALSE(legate::is_running_in_task()); }

TEST_F(IsRunningInTask, UserThread)
{
  std::thread thd{[] { EXPECT_FALSE(legate::is_running_in_task()); }};
  thd.join();
}

TEST_F(IsRunningInTask, InSingleTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  runtime->submit(runtime->create_task(library, Checker::TASK_ID));
}

TEST_F(IsRunningInTask, InIndexTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  runtime->submit(
    runtime->create_task(library, Checker::TASK_ID, legate::tuple<std::uint64_t>{2, 2}));
}

using IsRunningInTaskNoRuntime = ::testing::Test;

TEST_F(IsRunningInTaskNoRuntime, BeforeInit) { EXPECT_FALSE(legate::is_running_in_task()); }

TEST_F(IsRunningInTaskNoRuntime, UserThread)
{
  std::thread thd{[] { EXPECT_FALSE(legate::is_running_in_task()); }};
  thd.join();
}

}  // namespace test_is_running_in_task
