/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace exception {

using Exception = DefaultFixture;

const char* library_name = "test_exception";

static legate::Logger logger(library_name);

enum TaskIDs {
  EXCEPTION_TASK = 0,
};

const char* EXN_MSG   = "Exception Test";
constexpr size_t SIZE = 10;

struct ExceptionTask : public legate::LegateTask<ExceptionTask> {
  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE(context.can_raise_exception());
    auto raise = context.scalar(0).value<bool>();
    auto index = context.scalar(1).value<int32_t>();
    // Make sure only some of the  point tasks raise an exception
    if (raise && (context.is_single_task() || context.get_task_index()[0] == 0)) {
      if (context.is_single_task())
        logger.debug() << "Raise an exception for index " << index;
      else
        logger.debug() << "Raise an exception for index " << index << " (task "
                       << context.get_task_index() << ")";
      throw legate::TaskException(index, EXN_MSG);
    } else {
      if (context.is_single_task())
        logger.debug() << "Don't raise an exception for index " << index;
      else
        logger.debug() << "Don't raise an exception for index " << index << " (task "
                       << context.get_task_index() << ")";
      ;
    }
  }
};

void prepare()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  ExceptionTask::register_variants(context, EXCEPTION_TASK);
}

void test_single()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto issue_task = [&](int32_t index, bool raise) {
    auto task = runtime->create_task(context, EXCEPTION_TASK);
    task.add_scalar_arg(legate::Scalar(raise));
    task.add_scalar_arg(legate::Scalar(index));
    task.throws_exception(true);
    runtime->submit(std::move(task));
  };

  EXPECT_THROW(issue_task(0, true), legate::TaskException);
}

TEST_F(Exception, Single)
{
  prepare();

  test_single();
}

}  // namespace exception
