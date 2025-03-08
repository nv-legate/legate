/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>
#include <vector>

namespace premature_free_test {

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

namespace {

constexpr std::uint64_t EXT = 17;

struct Deleter {
  explicit Deleter(bool* target) : deleted{target} {}
  void operator()(void* /*ptr*/) const { *deleted = true; }
  bool* deleted{};
};

}  // namespace

TEST_F(PrematureFree, ArrayFromTmpStore)
{
  auto vec     = std::vector<std::int64_t>(EXT);
  auto deleted = false;
  auto runtime = legate::Runtime::get_runtime();

  {
    auto alloc = legate::ExternalAllocation::create_sysmem(
      vec.data(), EXT * sizeof(std::int64_t), false, Deleter{&deleted});
    auto arr = legate::LogicalArray{runtime->create_store({EXT}, legate::int64(), alloc)};
    static_cast<void>(runtime->create_store({EXT}, legate::int64()));
    runtime->issue_execution_fence(true);

    EXPECT_FALSE(deleted);
  }

  // We have two free fields of shape {EXT} and type int64, so we need to create two stores to
  // have the deleter invoked
  auto st1 = runtime->create_store({EXT}, legate::int64());
  auto st2 = runtime->create_store({EXT}, legate::int64());
  runtime->issue_execution_fence(true);

  EXPECT_TRUE(deleted);
}

TEST_F(PrematureFree, NullableArrayFromTmpStores)
{
  auto vec1     = std::vector<std::int64_t>(EXT);
  auto vec2     = std::vector<std::uint8_t>(EXT);
  auto deleted1 = false;
  auto deleted2 = false;
  auto runtime  = legate::Runtime::get_runtime();

  {
    auto alloc1 = legate::ExternalAllocation::create_sysmem(
      vec1.data(), EXT * sizeof(std::int64_t), false, Deleter{&deleted1});
    auto alloc2 = legate::ExternalAllocation::create_sysmem(
      vec2.data(), EXT * sizeof(std::uint8_t), false, Deleter{&deleted2});
    auto arr = legate::LogicalArray{runtime->create_store({EXT}, legate::int64(), alloc1),
                                    runtime->create_store({EXT}, legate::bool_(), alloc2)};
    static_cast<void>(runtime->create_store({EXT}, legate::int64()));
    static_cast<void>(runtime->create_store({EXT}, legate::bool_()));
    runtime->issue_execution_fence(true);

    EXPECT_FALSE(deleted1);
    EXPECT_FALSE(deleted2);
  }

  // We have four free fields, two of shape {EXT} and type int64 and another two of shape {EXT} and
  // type bool, so we need to create two stores for each spec to have both deleters invoked
  auto st1 = runtime->create_store({EXT}, legate::int64());
  auto st2 = runtime->create_store({EXT}, legate::int64());
  auto st3 = runtime->create_store({EXT}, legate::bool_());
  auto st4 = runtime->create_store({EXT}, legate::bool_());
  runtime->issue_execution_fence(true);

  EXPECT_TRUE(deleted1);
  EXPECT_TRUE(deleted2);
}

TEST_F(PrematureFree, ArrayMovedToAutoTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto arr = runtime->create_array({EXT}, legate::int64());
  runtime->issue_fill(arr, legate::Scalar{std::int64_t{0}});

  auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
  task.add_input(arr);
  // the old array should still be alive until the task is done
  arr = runtime->create_array({EXT}, legate::int64());
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

TEST_F(PrematureFree, ArrayMovedToManualTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto arr = runtime->create_store({EXT}, legate::int64());
  runtime->issue_fill(arr, legate::Scalar{std::int64_t{0}});

  auto task = runtime->create_task(
    library, DummyTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{2, 4});
  task.add_input(arr);
  // the old store should still be alive until the task is done
  arr = runtime->create_store({EXT}, legate::int64());
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

}  // namespace premature_free_test
