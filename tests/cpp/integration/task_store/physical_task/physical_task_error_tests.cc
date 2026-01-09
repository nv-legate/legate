/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_array.h>
#include <legate/operation/detail/task.h>

#include <integration/task_store/task_common.h>

namespace test_task_store {

namespace {

constexpr std::int32_t ERROR_TEST_BASE_TASK_ID = 0;

}  // namespace

// Task that throws an exception for testing error propagation
struct ExceptionThrowingTask : public legate::LegateTask<ExceptionThrowingTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{
      legate::LocalTaskID{ERROR_TEST_BASE_TASK_ID + 1}};  // Use independent task ID

  static void cpu_variant(legate::TaskContext /* context */)
  {
    throw std::runtime_error{"PhysicalTask exception test"};
  }
};

// Task that has NO variants - will always cause variant mismatch
struct NoVariantTask : public legate::LegateTask<NoVariantTask> {
  static inline const auto TASK_CONFIG =                               // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ERROR_TEST_BASE_TASK_ID}};  // Use independent task ID
};

// Config class for error tests with independent task registration
class ErrorTestConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_physical_task_errors";

  static void registration_callback(legate::Library library)
  {
    ExceptionThrowingTask::register_variants(library);
    NoVariantTask::register_variants(library);
  }
};

// Test fixture for error tests using RegisterOnceFixture pattern
class PhysicalTaskErrorTests : public RegisterOnceFixture<ErrorTestConfig> {};

// Test invalid task ID handling
TEST_F(TaskStoreTests, PhysicalTaskInvalidTaskID)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  constexpr auto INVALID_TASK_ID = 9999;
  auto invalid_task_id           = legate::LocalTaskID{INVALID_TASK_ID};

  ASSERT_THAT([&]() { static_cast<void>(runtime->create_physical_task(library, invalid_task_id)); },
              testing::ThrowsMessage<std::out_of_range>(testing::HasSubstr("does not have task")));
}

// Test invalid library handling
TEST_F(TaskStoreTests, PhysicalTaskInvalidLibrary)
{
  auto runtime = legate::Runtime::get_runtime();

  ASSERT_THAT(
    [&]() {
      auto invalid_library = runtime->find_library("NonExistentLibrary");
      static_cast<void>(runtime->create_physical_task(invalid_library, legate::LocalTaskID{0}));
    },
    testing::ThrowsMessage<std::out_of_range>(testing::HasSubstr("Library")));
}

// Test exception propagation from PhysicalTask
// PhysicalTask does not support graceful exception handling and will abort the process
// when an exception occurs within a task variant. This is verified using a death test.
TEST_F(PhysicalTaskErrorTests, PhysicalTaskExceptionPropagationDeathTest)
{
  ASSERT_DEATH(
    {
      auto runtime = legate::Runtime::get_runtime();
      auto library = runtime->find_library(ErrorTestConfig::LIBRARY_NAME);

      auto physical_task =
        runtime->create_physical_task(library, ExceptionThrowingTask::TASK_CONFIG.task_id());

      runtime->submit(std::move(physical_task));
    },
    "PhysicalTask exception test");
}

// Test that creating a task with no registered variants throws an exception
TEST_F(PhysicalTaskErrorTests, PhysicalTaskNoVariantError)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(ErrorTestConfig::LIBRARY_NAME);

  ASSERT_THAT(
    [&]() {
      static_cast<void>(
        runtime->create_physical_task(library, NoVariantTask::TASK_CONFIG.task_id()));
    },
    testing::ThrowsMessage<std::invalid_argument>(testing::AllOf(
      testing::HasSubstr("does not have any valid variant"), testing::HasSubstr("NoVariantTask"))));
}

}  // namespace test_task_store
