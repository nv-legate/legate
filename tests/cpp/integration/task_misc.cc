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

namespace task_misc {

namespace {

constexpr std::int32_t SCL_VAL = 42;

}  // namespace

class NormalTask : public legate::LegateTask<NormalTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};
  static void cpu_variant(legate::TaskContext context)
  {
    auto scalar_value = context.scalar(0).value<std::int32_t>();
    ASSERT_EQ(scalar_value, SCL_VAL);
  }
};

class ExceptionUnboundTask : public legate::LegateTask<ExceptionUnboundTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};
  static void cpu_variant(legate::TaskContext context)
  {
    auto scalar_value = context.scalar(0).value<std::int32_t>();
    ASSERT_EQ(scalar_value, SCL_VAL);
    ASSERT_TRUE(context.can_raise_exception());

    auto store = context.output(0).data();
    store.bind_empty_data();
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_misc";
  static void registration_callback(legate::Library library)
  {
    NormalTask::register_variants(library);
    ExceptionUnboundTask::register_variants(library);
  }
};

class TaskMiscTest : public RegisterOnceFixture<Config> {};

legate::AutoTask create_auto_scalar_out_red()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, NormalTask::TASK_ID);

  constexpr std::int32_t value             = 10;
  auto store_out                           = runtime->create_store(legate::Scalar{value});
  constexpr legate::ReductionOpKind red_op = legate::ReductionOpKind::ADD;
  auto store_red                           = runtime->create_store(legate::Scalar{value});

  task.add_output(store_out);
  task.add_reduction(store_red, red_op);
  task.add_scalar_arg(SCL_VAL);

  return task;
}

legate::ManualTask create_manual_exception_unbound()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    library, ExceptionUnboundTask::TASK_ID, legate::tuple<std::uint64_t>{4, 2});
  auto store = runtime->create_store(legate::int64(), 2);

  task.throws_exception(true);
  task.add_output(store);
  task.add_scalar_arg(SCL_VAL);

  return task;
}

void test_auto_scalar_out_red()
{
  auto task = create_auto_scalar_out_red();
  legate::Runtime::get_runtime()->submit(std::move(task));
}

void test_manual_exception_unbound()
{
  auto task = create_manual_exception_unbound();
  legate::Runtime::get_runtime()->submit(std::move(task));
}

TEST_F(TaskMiscTest, AutoScalarOutRed) { test_auto_scalar_out_red(); }

TEST_F(TaskMiscTest, ManualExecptionUnbound) { test_manual_exception_unbound(); }

}  // namespace task_misc
