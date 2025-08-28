/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_find_memory_kind {

struct Checker : public legate::LegateTask<Checker> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

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

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_is_running_in_task";

  static void registration_callback(legate::Library library)
  {
    Checker::register_variants(library);
  }
};

class FindMemoryKind : public RegisterOnceFixture<Config> {};

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

TEST_F(FindMemoryKind, InSingleTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  runtime->submit(runtime->create_task(library, Checker::TASK_CONFIG.task_id()));
}

TEST_F(FindMemoryKind, InIndexTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  runtime->submit(runtime->create_task(
    library, Checker::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{2, 2}));
}

}  // namespace test_find_memory_kind
