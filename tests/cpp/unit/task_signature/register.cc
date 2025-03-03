/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/partitioning/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/task/task_signature.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <utilities/utilities.h>

namespace test_task_signature_register {

namespace {

[[nodiscard]] legate::LogicalArray make_array()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_array(legate::Shape{3}, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{0});
  return ret;
}

class BasicTask : public legate::LegateTask<BasicTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_SIGNATURE = legate::TaskSignature{}.inputs(1).outputs(1).scalars(1);

  static void cpu_variant(legate::TaskContext) {}
};

class RangeTask : public legate::LegateTask<RangeTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_SIGNATURE =
    legate::TaskSignature{}.inputs(1, 2).outputs(1, 2).redops(1, 2).scalars(1, 2);

  static void cpu_variant(legate::TaskContext) {}
};

class ConstrainedTask : public legate::LegateTask<ConstrainedTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{2};

  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_SIGNATURE =
    legate::TaskSignature{}.inputs(1).outputs(1).constraints(
      {{legate::align(legate::proxy::inputs[0], legate::proxy::outputs[0])}});

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_register";

  static void registration_callback(legate::Library library)
  {
    BasicTask::register_variants(library);
    RangeTask::register_variants(library);
    ConstrainedTask::register_variants(library);
  }
};

}  // namespace

class TaskSignatureRegisterUnit : public RegisterOnceFixture<Config> {};

class TaskSignatureRegisterUnitBasicTask : public TaskSignatureRegisterUnit {
 public:
  void SetUp() override
  {
    TaskSignatureRegisterUnit::SetUp();

    auto* runtime = legate::Runtime::get_runtime();
    task          = std::make_unique<legate::AutoTask>(
      runtime->create_task(runtime->find_library(Config::LIBRARY_NAME), BasicTask::TASK_ID));
  }

  std::unique_ptr<legate::AutoTask> task;
};

TEST_F(TaskSignatureRegisterUnitBasicTask, Basic)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  // Argument checking occurs during submission
  runtime->submit(std::move(*task));
}

TEST_F(TaskSignatureRegisterUnitBasicTask, TooFewInputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  // Note, no input
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitBasicTask, TooManyInputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  // Note, too many inputs
  task->add_input(make_array());
  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitBasicTask, TooFewOutputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  // Note, no outputs
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitBasicTask, TooManyOutputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  // Note, too many outputs
  task->add_output(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitBasicTask, TooFewScalars)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  // Note, no scalars
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitBasicTask, TooManyScalars)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  // Note, too many scalars
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

class TaskSignatureRegisterUnitRangeTask : public TaskSignatureRegisterUnit {
 public:
  void SetUp() override
  {
    TaskSignatureRegisterUnit::SetUp();

    auto* runtime = legate::Runtime::get_runtime();
    task          = std::make_unique<legate::AutoTask>(
      runtime->create_task(runtime->find_library(Config::LIBRARY_NAME), RangeTask::TASK_ID));
  }

  std::unique_ptr<legate::AutoTask> task;
};

TEST_F(TaskSignatureRegisterUnitRangeTask, Basic)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  task->add_scalar_arg(legate::Scalar{0});
  runtime->submit(std::move(*task));
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooFewInputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  // Note, no inputs
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooManyInputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_input(make_array());
  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooFewOutputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  // Note, no outputs
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooManyOutputs)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  task->add_output(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooFewRedops)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  // Note, no redops
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooManyRedops)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooFewScalars)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  // Note, no scalars
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

TEST_F(TaskSignatureRegisterUnitRangeTask, TooManyScalars)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  task->add_reduction(make_array(), legate::ReductionOpKind::ADD);
  task->add_scalar_arg(legate::Scalar{0});
  task->add_scalar_arg(legate::Scalar{0});
  task->add_scalar_arg(legate::Scalar{0});
  ASSERT_THROW(runtime->submit(std::move(*task)), std::out_of_range);
}

class TaskSignatureRegisterUnitConstrainedTask : public TaskSignatureRegisterUnit {
 public:
  void SetUp() override
  {
    TaskSignatureRegisterUnit::SetUp();

    auto* runtime = legate::Runtime::get_runtime();

    task = std::make_unique<legate::AutoTask>(
      runtime->create_task(runtime->find_library(Config::LIBRARY_NAME), ConstrainedTask::TASK_ID));
  }

  std::unique_ptr<legate::AutoTask> task;
};

TEST_F(TaskSignatureRegisterUnitConstrainedTask, Basic)
{
  auto* runtime = legate::Runtime::get_runtime();

  task->add_input(make_array());
  task->add_output(make_array());
  runtime->submit(std::move(*task));
}

TEST_F(TaskSignatureRegisterUnitConstrainedTask, BadConstraint)
{
  auto in_var  = task->add_input(make_array());
  auto out_var = task->add_output(make_array());

  ASSERT_THROW(task->add_constraint(legate::align(in_var, out_var)), std::runtime_error);
}

// NOLINTBEGIN(cert-err58-cpp)
/// [Example task signature]
class ExampleTask : public legate::LegateTask<ExampleTask> {
 public:
  static inline const auto TASK_SIGNATURE =
    legate::TaskSignature{}  // The task expects exactly 2 inputs...
      .inputs(2)
      // But may take at least 3 and no more than 5 outputs...
      .outputs(3, 5)
      // While taking an unbounded number of scalars (but must have at least 1)
      .scalars(1, legate::TaskSignature::UNBOUNDED)
      // With the following constraints imposed on the arguments
      .constraints(
        {{// Align the first input with the first output
          legate::align(legate::proxy::inputs[0], legate::proxy::outputs[0]),
          // Broadcast ALL inputs
          legate::broadcast(legate::proxy::inputs),
          // All arguments (including axes) of constraints are supported
          legate::scale({1, 2, 3}, legate::proxy::outputs[1], legate::proxy::inputs[1])}});
};
/// [Example task signature]
// NOLINTEND(cert-err58-cpp)

namespace {

[[maybe_unused]] void dummy_example_function()
{
  std::ignore =
    /// [Align all inputs with output 0]
    legate::align(legate::proxy::inputs, legate::proxy::outputs[0])
    /// [Align all inputs with output 0]
    ;

  std::ignore =
    /// [Align all input 0 with output 1]
    legate::align(legate::proxy::inputs[0], legate::proxy::inputs[1])
    /// [Align all input 0 with output 1]
    ;

  std::ignore =
    /// [Broadcast input 0]
    legate::broadcast(legate::proxy::inputs[0])
    /// [Broadcast input 0]
    ;

  std::ignore =
    /// [Broadcast all outputs]
    legate::broadcast(legate::proxy::outputs)
    /// [Broadcast all outputs]
    ;
}

}  // namespace

}  // namespace test_task_signature_register
