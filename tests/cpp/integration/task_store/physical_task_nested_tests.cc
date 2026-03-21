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

constexpr std::int32_t NESTED_TEST_BASE_TASK_ID     = 0;
constexpr std::string_view NESTED_TEST_LIBRARY_NAME = "test_physical_task_nested";

struct ProcessorCoherenceVerifier : public legate::LegateTask<ProcessorCoherenceVerifier> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 0}};

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_EQ(context.num_scalars(), 2);

    auto parent_variant = static_cast<legate::VariantCode>(context.scalar(0).value<std::int32_t>());
    auto parent_proc_id = static_cast<std::uint64_t>(context.scalar(1).value<std::int64_t>());

    auto runtime         = legate::Runtime::get_runtime();
    auto current_variant = context.variant_kind();
    auto current_proc_id = runtime->get_executing_processor().id;

    ASSERT_EQ(current_variant, legate::VariantCode::CPU);
    ASSERT_EQ(parent_variant, legate::VariantCode::CPU);
    ASSERT_EQ(current_proc_id, parent_proc_id);
    ASSERT_TRUE(context.is_single_task());
  }
};

// AutoTask that creates nested task
struct AutoTaskWithPhysicalTask : public legate::LegateTask<AutoTaskWithPhysicalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 1}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(NESTED_TEST_LIBRARY_NAME);

    ASSERT_EQ(context.variant_kind(), legate::VariantCode::CPU);

    auto nested_task =
      runtime->create_task(library, ProcessorCoherenceVerifier::TASK_CONFIG.task_id());

    nested_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(context.variant_kind())});
    nested_task.add_scalar_arg(
      legate::Scalar{static_cast<std::int64_t>(runtime->get_executing_processor().id)});

    EXPECT_NO_THROW(runtime->submit(std::move(nested_task)));
  }
};

// ManualTask that creates nested AutoTask
struct ManualTaskWithPhysicalTask : public legate::LegateTask<ManualTaskWithPhysicalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 2}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(NESTED_TEST_LIBRARY_NAME);

    ASSERT_EQ(context.variant_kind(), legate::VariantCode::CPU);
    ASSERT_FALSE(context.is_single_task());
    ASSERT_FALSE(context.get_launch_domain().empty());

    auto nested_task =
      runtime->create_task(library, ProcessorCoherenceVerifier::TASK_CONFIG.task_id());

    nested_task.add_scalar_arg(legate::Scalar{static_cast<std::int32_t>(context.variant_kind())});
    nested_task.add_scalar_arg(
      legate::Scalar{static_cast<std::int64_t>(runtime->get_executing_processor().id)});

    EXPECT_NO_THROW(runtime->submit(std::move(nested_task)));
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

struct SimpleNegationTask : public legate::LegateTask<SimpleNegationTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 4}};

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_EQ(context.inputs().size(), 1);
    ASSERT_EQ(context.outputs().size(), 1);

    auto input_array  = context.input(0);
    auto output_array = context.output(0);

    auto input_store  = input_array.data();
    auto output_store = output_array.data();

    auto input_accessor  = input_store.read_accessor<std::int32_t, 1>();
    auto output_accessor = output_store.write_accessor<std::int32_t, 1>();

    auto domain = input_store.domain();
    for (legate::PointInRectIterator<1> it{domain}; it.valid(); ++it) {
      auto point             = *it;
      output_accessor[point] = -input_accessor[point];
    }
  }
};

struct ParentTaskWithNestedAutoTask : public legate::LegateTask<ParentTaskWithNestedAutoTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{NESTED_TEST_BASE_TASK_ID + 5}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library(NESTED_TEST_LIBRARY_NAME);

    ASSERT_EQ(context.inputs().size(), 1);
    ASSERT_EQ(context.outputs().size(), 1);

    auto input_physical_array  = context.input(0);
    auto output_physical_array = context.output(0);

    auto input_logical_store  = input_physical_array.data().to_logical_store();
    auto output_logical_store = output_physical_array.data().to_logical_store();

    auto input_logical_array  = legate::LogicalArray{input_logical_store};
    auto output_logical_array = legate::LogicalArray{output_logical_store};

    auto auto_task = runtime->create_task(library, SimpleNegationTask::TASK_CONFIG.task_id());
    auto_task.add_input(input_logical_array);
    auto_task.add_output(output_logical_array);

    runtime->submit(std::move(auto_task));
  }
};

class NestedTestConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = NESTED_TEST_LIBRARY_NAME;

  static void registration_callback(legate::Library library)
  {
    ProcessorCoherenceVerifier::register_variants(library);
    AutoTaskWithPhysicalTask::register_variants(library);
    ManualTaskWithPhysicalTask::register_variants(library);
    PrimeSquareTask::register_variants(library);
    SimpleNegationTask::register_variants(library);
    ParentTaskWithNestedAutoTask::register_variants(library);
  }
};

class NestedTaskTests : public RegisterOnceFixture<NestedTestConfig> {};

}  // namespace

TEST_F(NestedTaskTests, FromAutoTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(NestedTestConfig::LIBRARY_NAME);

  runtime->submit(runtime->create_task(library, AutoTaskWithPhysicalTask::TASK_CONFIG.task_id()));
  runtime->issue_execution_fence(true);
}

TEST_F(NestedTaskTests, FromManualTask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(NestedTestConfig::LIBRARY_NAME);

  auto launch_domain = legate::detail::to_domain(legate::Span<const std::uint64_t>{{2, 2}});
  auto manual_task =
    runtime->create_task(library, ManualTaskWithPhysicalTask::TASK_CONFIG.task_id(), launch_domain);
  runtime->submit(std::move(manual_task));
  runtime->issue_execution_fence(true);
}

TEST_F(NestedTaskTests, InlineExecution)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(NestedTestConfig::LIBRARY_NAME);

  constexpr std::int32_t SIZE = 10;
  auto input_store            = runtime->create_store(legate::Shape{SIZE}, legate::int32());
  {
    auto input_physical_store = input_store.get_physical_store();
    auto input_accessor       = input_physical_store.write_accessor<std::int32_t, 1>();
    for (std::int32_t i = 0; i < SIZE; ++i) {
      input_accessor[legate::Point<1>{i}] = i + 1;
    }
  }

  auto output_store = runtime->create_store(legate::Shape{SIZE}, legate::int32());

  auto parent_task =
    runtime->create_task(library, ParentTaskWithNestedAutoTask::TASK_CONFIG.task_id());
  parent_task.add_input(legate::LogicalArray{input_store});
  parent_task.add_output(output_store);

  runtime->submit(std::move(parent_task));
  runtime->issue_execution_fence(true);
  {
    auto output_physical_store = output_store.get_physical_store();
    auto output_accessor       = output_physical_store.read_accessor<std::int32_t, 1>();
    for (std::int32_t i = 0; i < SIZE; ++i) {
      auto expected = -(i + 1);
      auto actual   = output_accessor[legate::Point<1>{i}];
      ASSERT_EQ(actual, expected) << "Mismatch at index " << i;
    }
  }
}

TEST_F(NestedTaskTests, CreateLogicalStoreFromPhysical)
{
  auto runtime = legate::Runtime::get_runtime();

  auto store    = runtime->create_store(legate::Shape{5}, legate::int32());
  auto physical = store.get_physical_store();

  auto logical = physical.to_logical_store();

  ASSERT_EQ(logical.dim(), 1);
  ASSERT_EQ(logical.extents()[0], 5);
  ASSERT_EQ(logical.type(), legate::int32());
}

TEST_F(NestedTaskTests, CreateLogicalStoreFromPhysicalMultiDim)
{
  auto runtime = legate::Runtime::get_runtime();

  auto store_2d    = runtime->create_store(legate::Shape{3, 4}, legate::float32());
  auto physical_2d = store_2d.get_physical_store();
  auto logical_2d  = physical_2d.to_logical_store();

  ASSERT_EQ(logical_2d.dim(), 2);
  ASSERT_EQ(logical_2d.extents()[0], 3);
  ASSERT_EQ(logical_2d.extents()[1], 4);

  auto store_3d    = runtime->create_store(legate::Shape{2, 3, 4}, legate::int64());
  auto physical_3d = store_3d.get_physical_store();
  auto logical_3d  = physical_3d.to_logical_store();

  ASSERT_EQ(logical_3d.dim(), 3);
  ASSERT_EQ(logical_3d.extents()[0], 2);
  ASSERT_EQ(logical_3d.extents()[1], 3);
  ASSERT_EQ(logical_3d.extents()[2], 4);
}

TEST_F(NestedTaskTests, CreateLogicalStoreFromPhysicalRejectsFuture)
{
  auto runtime = legate::Runtime::get_runtime();

  auto scalar_store =
    runtime->create_store(legate::Shape{1}, legate::int32(), /*optimize_scalar=*/true);
  auto physical = scalar_store.get_physical_store();

  EXPECT_THROW((void)physical.to_logical_store(), std::runtime_error);
}

TEST_F(NestedTaskTests, LogicalStoreFromPhysicalRejectsTransform)
{
  auto runtime = legate::Runtime::get_runtime();

  constexpr std::int64_t test_dim = 10;
  auto store    = runtime->create_store(legate::Shape{test_dim, test_dim}, legate::int32());
  auto physical = store.get_physical_store();
  auto logical  = physical.to_logical_store();

  EXPECT_THROW((void)logical.slice(0, legate::Slice{2, 5}), std::runtime_error);
  EXPECT_THROW((void)logical.transpose({1, 0}), std::runtime_error);
  EXPECT_THROW((void)logical.promote(0, 5), std::runtime_error);
  EXPECT_THROW((void)logical.project(0, 0), std::runtime_error);
  EXPECT_THROW((void)logical.delinearize(0, {2, 5}), std::runtime_error);
}

TEST_F(NestedTaskTests, LogicalStorePointsToSamePhysical)
{
  auto runtime = legate::Runtime::get_runtime();

  auto store           = runtime->create_store(legate::Shape{5, 5}, legate::int32());
  auto input_physical  = store.get_physical_store();
  auto logical         = input_physical.to_logical_store();
  auto output_physical = logical.get_physical_store();

  EXPECT_EQ(input_physical.impl().get(), output_physical.impl().get());
}

}  // namespace test_task_store
