/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/detail/env.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace oom_message {

namespace {

constexpr auto SMALL_SIZE    = 100;
constexpr auto HUGE_EXPONENT = 50;
constexpr auto HUGE_SIZE     = static_cast<std::uint64_t>(1) << HUGE_EXPONENT;  // 1 PiB

class DummyTask : public legate::LegateTask<DummyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_oom_message";

  static void registration_callback(legate::Library library)
  {
    DummyTask::register_variants(library);
  }
};

class OomMessageDeathTest : public RegisterOnceFixture<Config> {};

void test_oom_message()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // Declare stores (none get initialized until used)
  std::vector<legate::LogicalStore> stores;
  {
    const legate::Scope scope{"uninitialized_store_decl"};

    stores.emplace_back(runtime->create_store(legate::Shape{SMALL_SIZE}, legate::int8()));
    runtime->issue_fill(stores.back(), legate::Scalar{std::int8_t{0}});
  }
  {
    const legate::Scope scope{"small_store_decl"};

    stores.emplace_back(runtime->create_store(legate::Shape{SMALL_SIZE}, legate::int8()));
    runtime->issue_fill(stores.back(), legate::Scalar{std::int8_t{0}});
  }
  {
    const legate::Scope scope{"huge_store_decl"};

    stores.emplace_back(runtime->create_store(legate::Shape{HUGE_SIZE}, legate::int8()));
    runtime->issue_fill(stores.back(), legate::Scalar{std::int8_t{0}});
  }

  // Launch tasks
  {
    const legate::Scope scope{"launch_using_small_store_part"};
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());

    task.add_output(stores.at(1).slice(0, legate::Slice{1}));
    runtime->submit(std::move(task));
    runtime->issue_execution_fence(/*block=*/true);
  }
  {
    const legate::Scope scope{"launch_using_small_store_full"};
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());

    task.add_input(stores.at(1));
    runtime->submit(std::move(task));
    runtime->issue_execution_fence(/*block=*/true);
  }
  {
    const legate::Scope scope{"launch_using_huge_store"};
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());

    task.add_input(stores.at(1));
    task.add_output(stores.at(2));
    runtime->submit(std::move(task));
    runtime->issue_execution_fence(/*block=*/true);
  }
}

}  // namespace

TEST_F(OomMessageDeathTest, Simple)
{
  auto runtime = legate::Runtime::get_runtime();

  if (runtime->get_machine().count() > 1) {
    GTEST_SKIP() << "Test requires exactly 1 processor";
  }

  if (legate::detail::experimental::LEGATE_INLINE_TASK_LAUNCH.get(/* default_value =*/false)) {
    GTEST_SKIP() << "Test requires Legion task launch";
  }

  EXPECT_DEATH(
    test_oom_message(),
    // TODO(mpapadakis): Matching against multi-line regexes appears to be hit-or-miss, so we just
    // check each expected line in isolation.
    ::testing::AllOf(
      ::testing::ContainsRegex("Failed to allocate .*DummyTask\\[launch_using_huge_store\\]"),
      ::testing::ContainsRegex("corresponding to a LogicalStore allocated at huge_store_decl"),
      ::testing::Not(
        ::testing::ContainsRegex("LogicalStore allocated at uninitialized_store_decl")),
      ::testing::ContainsRegex("LogicalStore allocated at small_store_decl"),
      ::testing::ContainsRegex("Instance .* of size 100 covering elements <0>\\.\\.<99>.*"),
      ::testing::ContainsRegex(
        "created for an operation launched at launch_using_small_store_full"),
      ::testing::ContainsRegex("Instance .* of size 99 covering elements <1>\\.\\.<99>.*"),
      ::testing::ContainsRegex(
        "created for an operation launched at launch_using_small_store_part")));
}

}  // namespace oom_message
