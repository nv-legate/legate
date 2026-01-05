/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_array.h>
#include <legate/operation/detail/task.h>
#include <legate/utilities/detail/tuple.h>

#include <cmath>
#include <integration/task_store/task_common.h>

namespace test_task_store {

namespace {

constexpr std::int32_t CONTEXT_TEST_BASE_TASK_ID     = 0;
constexpr std::string_view CONTEXT_TEST_LIBRARY_NAME = "test_physical_task_context";

}  // namespace

// Custom task for testing mixed I/O: reads from input, modifies, writes to output
struct MixedIOTask : public legate::LegateTask<MixedIOTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CONTEXT_TEST_BASE_TASK_ID + 0}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    constexpr auto MIXED_IO_INCREMENT = 10;  // Value added in mixed I/O operation

    // Get input and output arrays (they can be the same PhysicalArray)
    const legate::PhysicalArray input_array  = context.input(0);
    const legate::PhysicalArray output_array = context.output(0);

    const legate::PhysicalStore input  = input_array.data();
    const legate::PhysicalStore output = output_array.data();

    const auto shape = input.shape<2>();
    if (shape.empty()) {
      return;
    }

    // Read from input, modify, write to output (classic mixed I/O pattern)
    const auto input_acc  = input.read_accessor<std::int32_t, 2>(shape);
    const auto output_acc = output.write_accessor<std::int32_t, 2>(shape);

    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      // Read from input, add increment, write to output
      output_acc[*it] = input_acc[*it] + MIXED_IO_INCREMENT;
    }
  }
};

// Task that verifies PhysicalTask launch index is always [0] regardless of parent context
struct LaunchIndexVerifier : public legate::LegateTask<LaunchIndexVerifier> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CONTEXT_TEST_BASE_TASK_ID + 1}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    // Verify PhysicalTask always has launch index [0] and single task domain
    const auto& launch_index  = context.get_task_index();
    const auto& launch_domain = context.get_launch_domain();

    // PhysicalTask should always have zero launch index
    ASSERT_EQ(launch_index[0], 0);
    ASSERT_EQ(launch_index[1], 0);  // For 2D case

    // PhysicalTask should always be a single task
    ASSERT_TRUE(context.is_single_task());

    // Launch domain is hardcoded to [0,0] to [1,0] in InlineTaskContext (volume = 2)
    ASSERT_EQ(launch_domain.get_volume(), 2U);
    ASSERT_EQ(launch_domain.lo()[0], 0);
    ASSERT_EQ(launch_domain.lo()[1], 0);
    ASSERT_EQ(launch_domain.hi()[0], 1);  // hi is inclusive, so domain is [0,0] to [1,0]
    ASSERT_EQ(launch_domain.hi()[1], 0);
  }
};

// Parent task that creates PhysicalTask from within index launch to test nested behavior
struct NestedLaunchIndexTester : public legate::LegateTask<NestedLaunchIndexTester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CONTEXT_TEST_BASE_TASK_ID + 2}};  // Independent task ID

  static void cpu_variant(legate::TaskContext parent_context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(CONTEXT_TEST_LIBRARY_NAME);

    // Parent task has variable launch index from index launch
    ASSERT_FALSE(parent_context.is_single_task());  // Parent is from index launch

    // Create PhysicalTask from within this parent
    auto physical_task = runtime->create_physical_task(
      parent_context, library, LaunchIndexVerifier::TASK_CONFIG.task_id());

    // PhysicalTask will ALWAYS have [0,0] launch index regardless of parent's index
    runtime->submit(std::move(physical_task));
  }
};

// Config class for context tests with independent task registration
class ContextTestConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = CONTEXT_TEST_LIBRARY_NAME;

  static void registration_callback(legate::Library library)
  {
    MixedIOTask::register_variants(library);
    LaunchIndexVerifier::register_variants(library);
    NestedLaunchIndexTester::register_variants(library);
  }
};

// Test fixture for context tests using RegisterOnceFixture pattern
class PhysicalTaskContextTests : public RegisterOnceFixture<ContextTestConfig> {};

// Test mixed input/output: same PhysicalArray as both input and output
TEST_F(PhysicalTaskContextTests, PhysicalTaskMixedInputOutput)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(ContextTestConfig::LIBRARY_NAME);

  auto shape               = legate::Shape{{2, 2}};
  const auto initial_value = static_cast<std::int32_t>(42);
  auto logical_store       = runtime->create_store(shape, legate::int32());
  runtime->issue_fill(logical_store, legate::Scalar{initial_value});

  auto logical_array = legate::LogicalArray{logical_store};
  auto array         = logical_array.get_physical_array();

  auto task = runtime->create_physical_task(library, MixedIOTask::TASK_CONFIG.task_id());

  // Core test: same PhysicalArray as both input and output
  EXPECT_NO_THROW(task.add_input(array));
  EXPECT_NO_THROW(task.add_output(array));

  EXPECT_NO_THROW(runtime->submit(std::move(task)));

  // Verify: input (42) + 10 = output (52). No fence needed - PhysicalTask executes inline
  constexpr auto MIXED_IO_INCREMENT = 10;  // Must match value in MixedIOTask
  const auto expected_value         = initial_value + MIXED_IO_INCREMENT;
  dim_dispatch(/*dim=*/2, VerifyOutputBody{}, array.data(), expected_value);
}

// Test direct PhysicalTask creation has launch index [0,0]
TEST_F(PhysicalTaskContextTests, PhysicalTaskDirectLaunchIndexZero)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(ContextTestConfig::LIBRARY_NAME);

  auto physical_task =
    runtime->create_physical_task(library, LaunchIndexVerifier::TASK_CONFIG.task_id());

  runtime->submit(std::move(physical_task));
}

// Test PhysicalTask created from within index launch has launch index [0,0]
TEST_F(PhysicalTaskContextTests, PhysicalTaskNestedLaunchIndexZero)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(ContextTestConfig::LIBRARY_NAME);

  // Parent tasks run at [0,0], [0,1], [1,0], [1,1]; each creates PhysicalTask with launch_index
  // [0,0]
  auto launch_domain = legate::detail::to_domain(legate::Span<const std::uint64_t>{{2, 2}});
  auto manual_task =
    runtime->create_task(library, NestedLaunchIndexTester::TASK_CONFIG.task_id(), launch_domain);

  runtime->submit(std::move(manual_task));
}

// Test inline execution verification
TEST_F(TaskStoreTests, PhysicalTaskInlineExecution)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto shape               = legate::Shape{{2, 2}};
  const auto initial_value = static_cast<std::int32_t>(100);
  auto logical_store       = runtime->create_store(shape, legate::int32());
  runtime->issue_fill(logical_store, legate::Scalar{initial_value});

  auto logical_array = legate::LogicalArray{logical_store};
  auto array         = logical_array.get_physical_array();

  auto task = runtime->create_physical_task(library, legate::LocalTaskID{SIMPLE_TASK + 2});
  task.add_output(array);
  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{initial_value});

  runtime->submit(std::move(task));

  // Verify inline execution: if deferred, this would fail without execution_fence
  dim_dispatch(/*dim=*/2, VerifyOutputBody{}, array.data(), initial_value);
}

}  // namespace test_task_store
