/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>
#include <vector>

namespace premature_free_test {

namespace {

// This task is meant to do nothing
class DummyTask : public legate::LegateTask<DummyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_premature_free";

  static void registration_callback(legate::Library library)
  {
    DummyTask::register_variants(library);
  }
};

class PrematureFree : public RegisterOnceFixture<Config> {};

constexpr std::uint64_t EXT = 17;

struct Deleter {
  explicit Deleter(bool* target) : deleted{target} {}

  void operator()(void* /*ptr*/) const { *deleted = true; }

  bool* deleted{};
};

}  // namespace

TEST_F(PrematureFree, StoreMovedToAutoTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto st = runtime->create_store({EXT}, legate::int64());
  runtime->issue_fill(st, legate::Scalar{std::int64_t{0}});

  auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
  task.add_input(st);
  // the old array should still be alive until the task is done
  st = runtime->create_store({EXT}, legate::int64());
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

TEST_F(PrematureFree, StoreMovedToManualTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto st = runtime->create_store({EXT}, legate::int64());
  runtime->issue_fill(st, legate::Scalar{std::int64_t{0}});

  auto task = runtime->create_task(
    library, DummyTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{2, 4});
  task.add_input(st);
  // the old store should still be alive until the task is done
  st = runtime->create_store({EXT}, legate::int64());
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

}  // namespace premature_free_test
