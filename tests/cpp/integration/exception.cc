/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "legate.h"

namespace exception {

namespace {

const char* library_name = "test_exception";

static legate::Logger logger(library_name);

enum TaskIDs {
  EXCEPTION_TASK = 0,
};

const char* EXN_MSG   = "Exception Test";
constexpr size_t SIZE = 10;

struct ExceptionTask : public legate::LegateTask<ExceptionTask> {
  static void cpu_variant(legate::TaskContext& context)
  {
    EXPECT_TRUE(context.can_raise_exception());
    auto raise = context.scalars().at(0).value<bool>();
    auto index = context.scalars().at(1).value<int32_t>();
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

  // Test the immediate exception handling
  runtime->set_max_pending_exceptions(1);
  EXPECT_THROW(issue_task(0, true), legate::TaskException);

  // Increase the window size to test more interesting scenarios
  runtime->set_max_pending_exceptions(4);

  // These three tasks shouldn't fill upthe window
  issue_task(1, true);
  issue_task(2, false);
  issue_task(3, true);

  auto check_exn = [&](int32_t index) {
    auto exn = runtime->check_pending_task_exception();
    EXPECT_TRUE(exn.has_value());
    EXPECT_EQ(exn.value().index(), index);
    EXPECT_EQ(exn.value().error_message(), EXN_MSG);
  };
  // At this point, there are three pending exceptions and no outstanding exceptions
  // Querying the first one filters out empty ones and turns the remaining ones into outstanding
  // exceptions
  check_exn(1);
  // With this task, there will be one pending exception and one outstanding exceptions
  issue_task(4, true);
  // Querying the next puts the state back to one outstanding exception and no pending exception
  check_exn(3);
  issue_task(5, false);
  issue_task(6, false);
  // This task should hit the max pending exception limit and raise the exception
  EXPECT_THROW(issue_task(7, false), legate::TaskException);

  // At this point there's no outstanding or pending exception
  // So the query should return a null value
  EXPECT_FALSE(runtime->check_pending_task_exception().has_value());
  // And it should be idempotent
  EXPECT_FALSE(runtime->check_pending_task_exception().has_value());

  issue_task(7, true);
  issue_task(8, false);
  // Immediate check should raise the exception
  EXPECT_THROW(runtime->raise_pending_task_exception(), legate::TaskException);
  // Then nothing should happen after that and the operation should be idempotent
  runtime->raise_pending_task_exception();
  runtime->raise_pending_task_exception();

  issue_task(6, true);
  // Finally crank the maximum number of pending exceptions down to 1 will flush the window
  EXPECT_THROW(runtime->set_max_pending_exceptions(1), legate::TaskException);
}

void test_multi(bool use_auto_task)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  // Turn off immediate exception raise
  runtime->set_max_pending_exceptions(2);

  auto store = runtime->create_store({SIZE, SIZE}, legate::int64());
  if (use_auto_task) {
    auto task = runtime->create_task(context, EXCEPTION_TASK);
    auto part = task.declare_partition();
    // Dummy store argument to trigger parallelization
    task.add_output(store, part);
    task.throws_exception(true);
    task.add_scalar_arg(legate::Scalar(true));
    task.add_scalar_arg(legate::Scalar(12345));

    runtime->submit(std::move(task));
  } else {
    auto task = runtime->create_task(context, EXCEPTION_TASK, {2, 2});
    auto part = store.partition_by_tiling({SIZE / 2, SIZE / 2});
    // Dummy store argument to trigger parallelization
    task.add_output(part);
    task.throws_exception(true);
    task.add_scalar_arg(legate::Scalar(true));
    task.add_scalar_arg(legate::Scalar(12345));
    runtime->submit(std::move(task));
  }

  auto exn = runtime->check_pending_task_exception();
  EXPECT_TRUE(exn.has_value());
  EXPECT_EQ(exn.value().index(), 12345);
  EXPECT_EQ(exn.value().error_message(), EXN_MSG);
}

void test_pending()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  runtime->set_max_pending_exceptions(2);
  auto task = runtime->create_task(context, EXCEPTION_TASK);
  task.throws_exception(true);
  task.add_scalar_arg(legate::Scalar(false));
  task.add_scalar_arg(legate::Scalar(12345));

  runtime->submit(std::move(task));

  // Finish the test with a pending exception to check if the runtime cleans things up correctly
}

}  // namespace

TEST(Exception, Single)
{
  legate::Core::perform_registration<prepare>();

  test_single();
}

TEST(Exception, Multi)
{
  legate::Core::perform_registration<prepare>();

  test_multi(true /* use_auto_task */);
  test_multi(false /* use_auto_task */);
}

TEST(Exception, Pending)
{
  legate::Core::perform_registration<prepare>();

  test_pending();
}

}  // namespace exception
