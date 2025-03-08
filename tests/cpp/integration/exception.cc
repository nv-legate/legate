/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace exception {

namespace {

constexpr std::int32_t EXN_IDX   = 42;
constexpr std::uint32_t NUM_EXN  = 3;
constexpr std::uint32_t NUM_NORM = 7;

struct ExceptionTask : public legate::LegateTask<ExceptionTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};
  static void cpu_variant(legate::TaskContext context)
  {
    auto index = context.scalar(0).value<std::int32_t>();
    EXPECT_TRUE(context.can_raise_exception());
    // Make sure only some of the  point tasks raise an exception
    if (context.is_single_task() || context.get_task_index()[0] == 0) {
      throw legate::TaskException{index, "exception from the tester"};
    }
  }
};

struct NormalTask : public legate::LegateTask<NormalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_exception";
  static void registration_callback(legate::Library library)
  {
    ExceptionTask::register_variants(library);
    NormalTask::register_variants(library);
  }
};

class Exception : public RegisterOnceFixture<Config> {};

legate::AutoTask create_auto()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, ExceptionTask::TASK_CONFIG.task_id());
  task.throws_exception(true);
  task.add_scalar_arg(legate::Scalar{EXN_IDX});
  return task;
}

legate::ManualTask create_manual()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    library, ExceptionTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{4, 2});
  task.throws_exception(true);
  task.add_scalar_arg(legate::Scalar{EXN_IDX});
  return task;
}

void test_immediate_single()
{
  auto task = create_auto();
  try {
    legate::Runtime::get_runtime()->submit(std::move(task));
    FAIL();
  } catch (const legate::TaskException& exn) {
    EXPECT_EQ(exn.index(), EXN_IDX);
  }
}

void test_immediate_index()
{
  auto task = create_manual();
  try {
    legate::Runtime::get_runtime()->submit(std::move(task));
    FAIL();
  } catch (const legate::TaskException& exn) {
    EXPECT_EQ(exn.index(), EXN_IDX);
  }
}

void test_deferred_or_ignored(legate::ExceptionMode exception_mode)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(legate::Shape{4, 2}, legate::int64());

  {
    const legate::Scope scope{exception_mode};
    for (std::uint32_t idx = 0; idx < NUM_NORM; ++idx) {
      runtime->submit(runtime->create_task(library, NormalTask::TASK_CONFIG.task_id()));
    }
    for (std::uint32_t idx = 0; idx < NUM_EXN; ++idx) {
      auto task = create_auto();
      runtime->submit(std::move(task));
    }
    for (std::uint32_t idx = 0; idx < NUM_NORM; ++idx) {
      runtime->submit(runtime->create_task(library, NormalTask::TASK_CONFIG.task_id()));
    }
  }

  if (exception_mode == legate::ExceptionMode::IGNORED) {
    EXPECT_NO_THROW(runtime->raise_pending_exception());
    return;
  }

  try {
    runtime->raise_pending_exception();
    FAIL();
  } catch (const legate::TaskException& exn) {
    EXPECT_EQ(exn.index(), EXN_IDX);
  }
}

}  // namespace

TEST_F(Exception, ImmediateSingle) { test_immediate_single(); }

TEST_F(Exception, ImmediateIndex) { test_immediate_index(); }

TEST_F(Exception, Deferred) { test_deferred_or_ignored(legate::ExceptionMode::DEFERRED); }

TEST_F(Exception, Ignored) { test_deferred_or_ignored(legate::ExceptionMode::IGNORED); }

}  // namespace exception
