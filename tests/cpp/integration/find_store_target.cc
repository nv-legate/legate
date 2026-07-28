/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_find_store_target {

namespace {

struct Checker : public legate::LegateTask<Checker> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    EXPECT_EQ(legate::find_store_target_for_executing_processor(),
              legate::mapping::StoreTarget::SYSMEM);
    EXPECT_EQ(legate::find_store_target_for_executing_processor(true),
              legate::mapping::StoreTarget::SYSMEM);
    EXPECT_EQ(legate::find_store_target_for_executing_processor(false),
              legate::mapping::StoreTarget::SYSMEM);
  }

  static void gpu_variant(legate::TaskContext /*context*/)
  {
    EXPECT_EQ(legate::find_store_target_for_executing_processor(),
              legate::mapping::StoreTarget::ZCMEM);
    EXPECT_EQ(legate::find_store_target_for_executing_processor(true),
              legate::mapping::StoreTarget::ZCMEM);
    EXPECT_EQ(legate::find_store_target_for_executing_processor(false),
              legate::mapping::StoreTarget::FBMEM);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_find_store_target";

  static void registration_callback(legate::Library library)
  {
    Checker::register_variants(library);
  }
};

class FindStoreTarget : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(FindStoreTarget, Toplevel)
{
  // Control code is running on a CPU
  // FIXME(wonchanl): This can change once we start doing single-GPU acceleration
  EXPECT_EQ(legate::find_store_target_for_executing_processor(),
            legate::mapping::StoreTarget::SYSMEM);
  EXPECT_EQ(legate::find_store_target_for_executing_processor(true),
            legate::mapping::StoreTarget::SYSMEM);
  EXPECT_EQ(legate::find_store_target_for_executing_processor(false),
            legate::mapping::StoreTarget::SYSMEM);
}

TEST_F(FindStoreTarget, InSingleTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  runtime->submit(runtime->create_task(library, Checker::TASK_CONFIG.task_id()));
}

TEST_F(FindStoreTarget, InIndexTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  runtime->submit(runtime->create_task(
    library, Checker::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{2, 2}));
}

}  // namespace test_find_store_target
