/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_array.h>
#include <legate/operation/detail/task.h>

#include <cmath>
#include <integration/task_store/task_common.h>

namespace test_task_store {

namespace {

constexpr std::int32_t BASIC_TEST_BASE_TASK_ID     = 0;
constexpr std::string_view BASIC_TEST_LIBRARY_NAME = "test_physical_task_basic";

// PhysicalArray creation helper
legate::PhysicalArray make_physical_array(const legate::Shape& shape, bool optimize_scalar)
{
  auto* runtime      = legate::Runtime::get_runtime();
  auto store         = optimize_scalar ? runtime->create_store(shape, legate::int32(), true)
                                       : runtime->create_store(shape, legate::int32());
  auto logical_array = legate::LogicalArray{store};
  return logical_array.get_physical_array();
}

// PhysicalTask only supports normal (bound) arrays, no unbound arrays

// Task that tests creating PhysicalTask with TaskContext
struct TaskContextTester : public legate::LegateTask<TaskContextTester> {
  static inline const auto TASK_CONFIG =                                   // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BASIC_TEST_BASE_TASK_ID + 0}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(BASIC_TEST_LIBRARY_NAME);

    EXPECT_NO_THROW({
      auto task =
        runtime->create_physical_task(context, library, legate::LocalTaskID{SIMPLE_TASK + 2});
      static_cast<void>(task);
    });

    // Library-only overload must throw when called inside a task
    EXPECT_THROW(
      {
        auto task = runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});
        static_cast<void>(task);
      },
      std::invalid_argument);

    auto task_with_context =
      runtime->create_physical_task(context, library, legate::LocalTaskID{SIMPLE_TASK + 2});

    const auto test_value = static_cast<std::int32_t>(42);
    EXPECT_NO_THROW(task_with_context.add_scalar_arg(legate::Scalar{test_value}));
  }
};

// PhysicalTask test functions
void physical_task_normal_input(const legate::PhysicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // SimpleTask<2> is already registered globally via Config::registration_callback

  auto task =
    runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});  // Use 2D task

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1);

  // Use runtime->issue_fill() like AutoTask
  if (array.owner().has_value()) {
    runtime->issue_fill(array.owner().value(), legate::Scalar{in_value1});
  }

  // Test add_input method
  EXPECT_NO_THROW(task.add_input(array));
  task.add_scalar_arg(legate::Scalar{TaskDataMode::INPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void physical_task_normal_output(const legate::PhysicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // SimpleTask<2> is already registered globally via Config::registration_callback

  auto task =
    runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});  // Use 2D task

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1);

  // Test add_output method
  EXPECT_NO_THROW(task.add_output(array));
  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  // Verify values written by the task
  auto expected_value = in_value1;
  // Force synchronization - ensure PhysicalTask executes before verification
  runtime->issue_execution_fence(true);  // block=true waits for completion
  dim_dispatch(2, VerifyOutputBody{}, array.data(), expected_value);
}

void physical_task_normal_reduction(const legate::PhysicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});

  const auto in_value1  = static_cast<std::int32_t>(INT_VALUE1);
  constexpr auto red_op = legate::ReductionOpKind::ADD;

  // Use runtime->issue_fill() like AutoTask
  if (array.owner().has_value()) {
    runtime->issue_fill(array.owner().value(), legate::Scalar{static_cast<std::int32_t>(0)});
  }

  // Test add_reduction method
  EXPECT_NO_THROW(task.add_reduction(array, static_cast<std::int32_t>(red_op)));
  task.add_scalar_arg(legate::Scalar{TaskDataMode::REDUCTION});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  // Verify reduced values written by the task
  auto expected_value = in_value1;
  // Force synchronization - ensure PhysicalTask executes before verification
  runtime->issue_execution_fence(true);  // block=true waits for completion
  dim_dispatch(2, VerifyOutputBody{}, array.data(), expected_value);
}

// Simple API tests that just verify PhysicalTask methods can be called without exceptions
void physical_task_api_input_test(const legate::PhysicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // SimpleTask<2> is already registered globally via Config::registration_callback

  auto task =
    runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});  // Use 2D task

  // Test that add_input doesn't throw
  EXPECT_NO_THROW(task.add_input(array));

  // Don't submit the task - just test the API calls work
}

void physical_task_api_output_test(const legate::PhysicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // SimpleTask<2> is already registered globally via Config::registration_callback

  auto task =
    runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});  // Use 2D task

  // Test that add_output doesn't throw
  EXPECT_NO_THROW(task.add_output(array));

  // Don't submit the task - just test the API calls work
}

void physical_task_api_reduction_test(const legate::PhysicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // SimpleTask<2> is already registered globally via Config::registration_callback

  auto task =
    runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});  // Use 2D task

  constexpr auto red_op = legate::ReductionOpKind::ADD;

  // Test that add_reduction doesn't throw
  EXPECT_NO_THROW(task.add_reduction(array, static_cast<std::int32_t>(red_op)));

  // Don't submit the task - just test the API calls work
}

}  // namespace

// Test class definitions
class PhysicalTaskNormal : public TaskStoreTests,
                           public ::testing::WithParamInterface<std::tuple<bool, legate::Shape>> {};

class PhysicalTaskNormalInput : public PhysicalTaskNormal {};

class PhysicalTaskNormalOutput : public PhysicalTaskNormal {};

class PhysicalTaskNormalReduction : public PhysicalTaskNormal {};

// Simple API test classes
class PhysicalTaskAPI : public TaskStoreTests,
                        public ::testing::WithParamInterface<std::tuple<bool, legate::Shape>> {};

class PhysicalTaskAPIInput : public PhysicalTaskAPI {};

class PhysicalTaskAPIOutput : public PhysicalTaskAPI {};

class PhysicalTaskAPIReduction : public PhysicalTaskAPI {};

// Test instantiations
INSTANTIATE_TEST_SUITE_P(TaskStoreTests,
                         PhysicalTaskNormalInput,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(legate::Shape{{5, 5}},
                                                              legate::Shape{{3, 3}},
                                                              legate::Shape{{2, 4}})));

INSTANTIATE_TEST_SUITE_P(TaskStoreTests,
                         PhysicalTaskNormalOutput,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(legate::Shape{{5, 5}},
                                                              legate::Shape{{3, 3}},
                                                              legate::Shape{{2, 4}})));

INSTANTIATE_TEST_SUITE_P(TaskStoreTests,
                         PhysicalTaskNormalReduction,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(legate::Shape{{5, 5}},
                                                              legate::Shape{{3, 3}},
                                                              legate::Shape{{2, 4}})));

// API test instantiations - just test that the API calls work
INSTANTIATE_TEST_SUITE_P(TaskStoreTests,
                         PhysicalTaskAPIInput,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(legate::Shape{{2, 2}},
                                                              legate::Shape{{3, 3}})));

INSTANTIATE_TEST_SUITE_P(TaskStoreTests,
                         PhysicalTaskAPIOutput,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(legate::Shape{{2, 2}},
                                                              legate::Shape{{3, 3}})));

INSTANTIATE_TEST_SUITE_P(TaskStoreTests,
                         PhysicalTaskAPIReduction,
                         ::testing::Combine(::testing::Values(false, true),
                                            ::testing::Values(legate::Shape{{2, 2}},
                                                              legate::Shape{{3, 3}})));

// Test implementations
TEST_P(PhysicalTaskNormalInput, Basic)
{
  auto [optimize_scalar, shape] = GetParam();
  auto array                    = make_physical_array(shape, optimize_scalar);
  physical_task_normal_input(array);
}

TEST_P(PhysicalTaskNormalOutput, Basic)
{
  auto [optimize_scalar, shape] = GetParam();
  auto array                    = make_physical_array(shape, optimize_scalar);
  physical_task_normal_output(array);
}

TEST_P(PhysicalTaskNormalReduction, Basic)
{
  auto [optimize_scalar, shape] = GetParam();
  auto array                    = make_physical_array(shape, optimize_scalar);
  physical_task_normal_reduction(array);
}

// API tests - just verify that the API calls work without exceptions
TEST_P(PhysicalTaskAPIInput, Basic)
{
  auto [optimize_scalar, shape] = GetParam();
  auto array                    = make_physical_array(shape, optimize_scalar);
  physical_task_api_input_test(array);
}

TEST_P(PhysicalTaskAPIOutput, Basic)
{
  auto [optimize_scalar, shape] = GetParam();
  auto array                    = make_physical_array(shape, optimize_scalar);
  physical_task_api_output_test(array);
}

TEST_P(PhysicalTaskAPIReduction, Basic)
{
  auto [optimize_scalar, shape] = GetParam();
  auto array                    = make_physical_array(shape, optimize_scalar);
  physical_task_api_reduction_test(array);
}

// Config class for basic tests with independent task registration
class BasicTestConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = BASIC_TEST_LIBRARY_NAME;

  static void registration_callback(legate::Library library)
  {
    TaskContextTester::register_variants(library);
    // Register SimpleTask<2> - used by TaskContextTester for nested PhysicalTask testing
    SimpleTask<2>::register_variants(library);
  }
};

// Test fixture for basic tests using RegisterOnceFixture pattern
class PhysicalTaskBasicTests : public RegisterOnceFixture<BasicTestConfig> {};

// Test for TaskContext overload
TEST_F(PhysicalTaskBasicTests, PhysicalTaskWithTaskContext)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(BasicTestConfig::LIBRARY_NAME);

  // Submit a task that will test the TaskContext overload from within a task execution context
  runtime->submit(runtime->create_task(library, TaskContextTester::TASK_CONFIG.task_id()));
}

}  // namespace test_task_store
