/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION &
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

constexpr std::int32_t NESTED_TEST_BASE_TASK_ID     = 0;
constexpr std::string_view NESTED_TEST_LIBRARY_NAME = "test_physical_task_nested";

}  // namespace

// Task that verifies machine and variant coherence between parent and PhysicalTask
struct MachineCoherenceVerifier : public legate::LegateTask<MachineCoherenceVerifier> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 0}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_EQ(context.num_scalars(), 4);

    auto parent_variant = static_cast<legate::VariantCode>(context.scalar(0).value<std::int32_t>());
    auto parent_cpu_count = static_cast<std::uint32_t>(context.scalar(1).value<std::int32_t>());
    auto parent_gpu_count = static_cast<std::uint32_t>(context.scalar(2).value<std::int32_t>());

    auto current_variant   = context.variant_kind();
    auto current_machine   = context.machine();
    auto current_cpu_count = current_machine.count(legate::mapping::TaskTarget::CPU);
    auto current_gpu_count = current_machine.count(legate::mapping::TaskTarget::GPU);
    auto current_is_single = context.is_single_task();

    ASSERT_EQ(current_variant, legate::VariantCode::CPU);
    ASSERT_EQ(current_cpu_count, parent_cpu_count);
    ASSERT_EQ(current_gpu_count, parent_gpu_count);
    ASSERT_TRUE(current_is_single);

    ASSERT_EQ(parent_variant, legate::VariantCode::CPU);
  }
};

// AutoTask that creates PhysicalTask from within its execution
struct AutoTaskWithPhysicalTask : public legate::LegateTask<AutoTaskWithPhysicalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 1}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(NESTED_TEST_LIBRARY_NAME);

    auto parent_variant   = context.variant_kind();
    auto parent_machine   = context.machine();
    auto parent_cpu_count = parent_machine.count(legate::mapping::TaskTarget::CPU);
    auto parent_gpu_count = parent_machine.count(legate::mapping::TaskTarget::GPU);
    auto parent_is_single = context.is_single_task();

    ASSERT_EQ(parent_variant, legate::VariantCode::CPU);
    ASSERT_GT(parent_cpu_count, 0U);

    auto physical_task = runtime->create_physical_task(
      context, library, MachineCoherenceVerifier::TASK_CONFIG.task_id());

    physical_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(parent_variant)});
    physical_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(parent_cpu_count)});
    physical_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(parent_gpu_count)});
    physical_task.add_scalar_arg(legate::Scalar{parent_is_single});

    EXPECT_NO_THROW(runtime->submit(std::move(physical_task)));
  }
};

// ManualTask that creates PhysicalTask from within its execution
struct ManualTaskWithPhysicalTask : public legate::LegateTask<ManualTaskWithPhysicalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 2}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(NESTED_TEST_LIBRARY_NAME);

    auto parent_variant   = context.variant_kind();
    auto parent_machine   = context.machine();
    auto parent_cpu_count = parent_machine.count(legate::mapping::TaskTarget::CPU);
    auto parent_gpu_count = parent_machine.count(legate::mapping::TaskTarget::GPU);
    auto parent_is_single = context.is_single_task();

    ASSERT_EQ(parent_variant, legate::VariantCode::CPU);
    ASSERT_GT(parent_cpu_count, 0U);
    ASSERT_FALSE(parent_is_single);
    ASSERT_FALSE(context.get_launch_domain().empty());

    auto physical_task = runtime->create_physical_task(
      context, library, MachineCoherenceVerifier::TASK_CONFIG.task_id());

    physical_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(parent_variant)});
    physical_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(parent_cpu_count)});
    physical_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(parent_gpu_count)});
    physical_task.add_scalar_arg(legate::Scalar{parent_is_single});

    EXPECT_NO_THROW(runtime->submit(std::move(physical_task)));
  }
};

// Task that computes square and square root of a prime number
struct PrimeSquareTask : public legate::LegateTask<PrimeSquareTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 3}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_EQ(context.inputs().size(), 1);
    ASSERT_EQ(context.outputs().size(), 2);

    auto input_array = context.input(0);
    auto input_store = input_array.data();

    auto square_output = context.output(0);
    auto sqrt_output   = context.output(1);

    auto input_accessor      = input_store.read_accessor<std::int32_t, 1>();
    const std::int32_t prime = input_accessor[legate::Point<1>{0}];

    const std::int32_t square = prime * prime;
    const float sqrt_val      = std::sqrt(static_cast<float>(prime));

    auto square_acc                 = square_output.data().write_accessor<std::int32_t, 1>();
    square_acc[legate::Point<1>{0}] = square;

    auto sqrt_acc                 = sqrt_output.data().write_accessor<float, 1>();
    sqrt_acc[legate::Point<1>{0}] = sqrt_val;
  }
};

// Nested AutoTask that creates PhysicalTask internally
struct NestedAutoTaskWithPhysicalTask : public legate::LegateTask<NestedAutoTaskWithPhysicalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 4}};  // Independent task ID

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(NESTED_TEST_LIBRARY_NAME);

    ASSERT_EQ(context.inputs().size(), 1);
    ASSERT_EQ(context.outputs().size(), 2);

    auto input_physical_array = context.input(0);
    auto square_output        = context.output(0);
    auto sqrt_output          = context.output(1);

    auto physical_task =
      runtime->create_physical_task(context, library, PrimeSquareTask::TASK_CONFIG.task_id());

    physical_task.add_input(input_physical_array);
    physical_task.add_output(square_output);
    physical_task.add_output(sqrt_output);

    physical_task.impl_()->add_scalar_output(square_output.data().impl());
    physical_task.impl_()->add_scalar_output(sqrt_output.data().impl());

    runtime->submit(std::move(physical_task));
  }
};

// Config class for nested tests with independent task registration
class NestedTestConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = NESTED_TEST_LIBRARY_NAME;

  static void registration_callback(legate::Library library)
  {
    MachineCoherenceVerifier::register_variants(library);
    AutoTaskWithPhysicalTask::register_variants(library);
    ManualTaskWithPhysicalTask::register_variants(library);
    PrimeSquareTask::register_variants(library);
    NestedAutoTaskWithPhysicalTask::register_variants(library);
  }
};

// Test fixture for nested tests using RegisterOnceFixture pattern
class PhysicalTaskNestedTests : public RegisterOnceFixture<NestedTestConfig> {};

// Test PhysicalTask creation from within AutoTask
TEST_F(PhysicalTaskNestedTests, PhysicalTaskFromAutoTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(NestedTestConfig::LIBRARY_NAME);

  runtime->submit(runtime->create_task(library, AutoTaskWithPhysicalTask::TASK_CONFIG.task_id()));
  runtime->issue_execution_fence(true);
}

// Test PhysicalTask creation from within ManualTask
TEST_F(PhysicalTaskNestedTests, PhysicalTaskFromManualTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(NestedTestConfig::LIBRARY_NAME);

  auto launch_domain = legate::detail::to_domain(legate::Span<const std::uint64_t>{{2, 2}});
  auto manual_task =
    runtime->create_task(library, ManualTaskWithPhysicalTask::TASK_CONFIG.task_id(), launch_domain);
  runtime->submit(std::move(manual_task));
  runtime->issue_execution_fence(true);
}

// Test multiple scalar outputs via nested AutoTask -> PhysicalTask
TEST_F(PhysicalTaskNestedTests, PhysicalTaskNestedInAutoTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(NestedTestConfig::LIBRARY_NAME);

  constexpr auto TEST_PRIME_NUMBER = 17;
  auto input_store                 = runtime->create_store(legate::Shape{1}, legate::int32());
  auto input_physical_store        = input_store.get_physical_store();
  {
    auto input_accessor                 = input_physical_store.write_accessor<std::int32_t, 1>();
    input_accessor[legate::Point<1>{0}] = TEST_PRIME_NUMBER;
  }

  auto square_store =
    runtime->create_store(legate::Shape{1}, legate::int32(), true /*optimize_scalar*/);
  auto sqrt_store =
    runtime->create_store(legate::Shape{1}, legate::float32(), true /*optimize_scalar*/);

  auto auto_task =
    runtime->create_task(library, NestedAutoTaskWithPhysicalTask::TASK_CONFIG.task_id());
  auto_task.add_input(legate::LogicalArray{input_store});
  auto_task.add_output(square_store);
  auto_task.add_output(sqrt_store);

  runtime->submit(std::move(auto_task));
  runtime->issue_execution_fence(true);

  {
    auto square_physical_store = square_store.get_physical_store();
    auto square_accessor       = square_physical_store.read_accessor<std::int32_t, 1>();
    auto computed_square       = square_accessor[legate::Point<1>{0}];
    ASSERT_EQ(computed_square, 289);

    auto sqrt_physical_store = sqrt_store.get_physical_store();
    auto sqrt_accessor       = sqrt_physical_store.read_accessor<float, 1>();
    auto computed_sqrt       = sqrt_accessor[legate::Point<1>{0}];
    ASSERT_NEAR(computed_sqrt, 4.123F, 0.001F);
  }
}

}  // namespace test_task_store
