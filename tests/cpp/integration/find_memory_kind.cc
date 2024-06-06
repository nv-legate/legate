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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace test_find_memory_kind {

using FindMemoryKind = DefaultFixture;

constexpr std::string_view LIBRARY_NAME = "test_is_running_in_task";

TEST_F(FindMemoryKind, Toplevel)
{
  // Control code is running on a CPU
  // FIXME(wonchanl): This can change once we start doing single-GPU acceleration
  EXPECT_EQ(legate::find_memory_kind_for_executing_processor(), legate::Memory::Kind::SYSTEM_MEM);
  EXPECT_EQ(legate::find_memory_kind_for_executing_processor(true),
            legate::Memory::Kind::SYSTEM_MEM);
  EXPECT_EQ(legate::find_memory_kind_for_executing_processor(false),
            legate::Memory::Kind::SYSTEM_MEM);
}

struct Checker : public legate::LegateTask<Checker> {
  static constexpr std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext /*context*/)
  {
    EXPECT_EQ(legate::find_memory_kind_for_executing_processor(), legate::Memory::Kind::SYSTEM_MEM);
    EXPECT_EQ(legate::find_memory_kind_for_executing_processor(true),
              legate::Memory::Kind::SYSTEM_MEM);
    EXPECT_EQ(legate::find_memory_kind_for_executing_processor(false),
              legate::Memory::Kind::SYSTEM_MEM);
  }
  static void gpu_variant(legate::TaskContext /*context*/)
  {
    EXPECT_EQ(legate::find_memory_kind_for_executing_processor(), legate::Memory::Kind::Z_COPY_MEM);
    EXPECT_EQ(legate::find_memory_kind_for_executing_processor(true),
              legate::Memory::Kind::Z_COPY_MEM);
    EXPECT_EQ(legate::find_memory_kind_for_executing_processor(false),
              legate::Memory::Kind::GPU_FB_MEM);
  }
};

TEST_F(FindMemoryKind, InSingleTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(LIBRARY_NAME);
  Checker::register_variants(library);

  runtime->submit(runtime->create_task(library, Checker::TASK_ID));
}

TEST_F(FindMemoryKind, InIndexTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(LIBRARY_NAME);
  Checker::register_variants(library);

  runtime->submit(
    runtime->create_task(library, Checker::TASK_ID, legate::tuple<std::uint64_t>{2, 2}));
}

using FindMemoryKindNoRuntime = ::testing::Test;

TEST_F(FindMemoryKindNoRuntime, BeforeInit)
{
  EXPECT_THROW(static_cast<void>(legate::find_memory_kind_for_executing_processor()),
               std::runtime_error);
}

}  // namespace test_find_memory_kind
