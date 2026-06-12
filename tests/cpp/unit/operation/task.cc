/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>
#include <utilities/utilities.h>

namespace operation_task_test {

namespace {

class ToStringTask : public legate::LegateTask<ToStringTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_operation_task";

  static void registration_callback(legate::Library library)
  {
    ToStringTask::register_variants(library);
  }
};

class TaskBaseUnit : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(TaskBaseUnit, SubmittedManualTaskCannotBeReused)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id(), {1});

  runtime->submit(std::move(task));

  // Accessors route through ManualTask::impl_(), which should reject descriptors that have already
  // been submitted.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_THAT([&] { static_cast<void>(task.provenance()); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Illegal to reuse task descriptors")));
}

TEST_F(TaskBaseUnit, SubmittedManualTaskCannotBeSubmittedAgain)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id(), {1});

  runtime->submit(std::move(task));

  // A repeated submit routes through ManualTask::release_(), so it must fail before returning a
  // null operation to the detail runtime.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_THAT([&] { runtime->submit(std::move(task)); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Illegal to reuse task descriptors")));
}

}  // namespace operation_task_test
