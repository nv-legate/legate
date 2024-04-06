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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace test_is_running_in_task {

using IsRunningInTask = DefaultFixture;

constexpr const char library_name[] = "test_is_running_in_task";

TEST_F(IsRunningInTask, Toplevel) { EXPECT_FALSE(legate::is_running_in_task()); }

struct Checker : public legate::LegateTask<Checker> {
  static constexpr std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext /*context*/)
  {
    EXPECT_TRUE(legate::is_running_in_task());
  }
};

TEST_F(IsRunningInTask, InSingleTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Checker::register_variants(library);

  runtime->submit(runtime->create_task(library, Checker::TASK_ID));
}

TEST_F(IsRunningInTask, InIndexTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Checker::register_variants(library);

  runtime->submit(
    runtime->create_task(library, Checker::TASK_ID, legate::tuple<std::uint64_t>{2, 2}));
}

using IsRunningInTaskNoRuntime = ::testing::Test;

TEST_F(IsRunningInTaskNoRuntime, BeforeInit) { EXPECT_FALSE(legate::is_running_in_task()); }

}  // namespace test_is_running_in_task
