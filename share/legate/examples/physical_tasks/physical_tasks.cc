/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <iostream>

namespace {

// Constants for the transformation
constexpr std::int64_t SOME_VALUE = 500;

// Simple physical task that performs transformation: ((x * 2) + 500) * 3
class SimplePhysicalTask : public legate::LegateTask<SimplePhysicalTask> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(
    legate::TaskSignature{}.inputs(1).outputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    std::cout << "  Physical Task: Performing transformation ((x * 2) + 500) * 3...\n";

    const legate::PhysicalArray input_array  = context.input(0);
    const legate::PhysicalArray output_array = context.output(0);

    const legate::PhysicalStore input = input_array.data();
    legate::PhysicalStore output      = output_array.data();

    const auto input_shape = input.shape<1>();
    const auto input_span  = input.span_read_accessor<std::int64_t, 1>(input_shape);
    const auto output_span = output.span_write_accessor<std::int64_t, 1>(input_shape);

    for (std::size_t i = 0; i < input_span.extent(0); ++i) {
      // Multi-step transformation: ((input * 2) + 500) * 3
      std::int64_t temp = input_span[i] * 2;  // Step 1
      temp += SOME_VALUE;                     // Step 2
      output_span[i] = temp * 3;              // Step 3
    }
  }
};

// Compound task that demonstrates AutoTask creating physical tasks internally
class CompoundTask : public legate::LegateTask<CompoundTask> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(
    legate::TaskSignature{}.inputs(1).outputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    std::cout << "=== AutoTask creating a single PhysicalTask internally ===\n";

    // Get input and output PhysicalArrays from the AutoTask context
    const legate::PhysicalArray input_array  = context.input(0);
    const legate::PhysicalArray output_array = context.output(0);

    // Display the input values (already initialized in main function)
    legate::PhysicalStore input_p_store = input_array.data();
    std::cout << "Input values: ";
    const auto input_read_span =
      input_p_store.span_read_accessor<std::int64_t, 1>(input_p_store.shape<1>());
    for (std::size_t i = 0; i < input_read_span.extent(0); ++i) {
      std::cout << input_read_span[i] << " ";
    }
    std::cout << "\n";

    // Get runtime and library for creating physical tasks
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->find_library("physical_tasks");

    std::cout << "Creating and submitting PhysicalTask from within AutoTask...\n";

    // Create and submit a single PhysicalTask
    std::cout << "  Creating PhysicalTask for transformation...\n";
    legate::PhysicalTask physical_task =
      runtime->create_physical_task(library, SimplePhysicalTask::TASK_CONFIG.task_id());
    physical_task.add_input(input_array);
    physical_task.add_output(output_array);
    runtime->submit(std::move(physical_task));

    std::cout << "PhysicalTask submitted successfully from within AutoTask!\n";
  }
};

void compound_autotask_with_physicaltasks()
{
  std::cout << "=== Demonstrating AutoTask allows creating physical tasks internally ===\n";

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library("physical_tasks");

  // Register task variants
  SimplePhysicalTask::register_variants(library);
  CompoundTask::register_variants(library);

  // Create input and output stores
  legate::LogicalStore input_store  = runtime->create_store(legate::Shape{10}, legate::int64());
  legate::LogicalStore output_store = runtime->create_store(legate::Shape{10}, legate::int64());

  // Initialize input data in main function (clean approach)
  legate::PhysicalStore input_p_store = input_store.get_physical_store();
  {
    const auto input_shape = input_p_store.shape<1>();
    const auto input_span  = input_p_store.span_write_accessor<std::int64_t, 1>(input_shape);
    for (std::size_t i = 0; i < input_span.extent(0); ++i) {
      input_span[i] =
        static_cast<std::int64_t>(i) + 10;  // Initialize with values: 10, 11, 12, ..., 19
    }
  }
  std::cout << "Input values (initialized before AutoTask): ";
  const auto input_read_span =
    input_p_store.span_read_accessor<std::int64_t, 1>(input_p_store.shape<1>());
  for (std::size_t i = 0; i < input_read_span.extent(0); ++i) {
    std::cout << input_read_span[i] << " ";
  }
  std::cout << "\n\n";

  // Create the top-level AutoTask that contains compound physical task internally
  std::cout << "Creating Top level AutoTask "
            << static_cast<std::int64_t>(CompoundTask::TASK_CONFIG.task_id()) << "\n";
  legate::AutoTask compound_auto_task =
    runtime->create_task(library, CompoundTask::TASK_CONFIG.task_id());

  // Add proper input and output to the AutoTask (clean, non-hacky approach)
  legate::LogicalArray input_logical  = legate::LogicalArray{input_store};
  legate::LogicalArray output_logical = legate::LogicalArray{output_store};

  compound_auto_task.add_input(input_logical);
  compound_auto_task.add_output(output_logical);

  // Submit the AutoTask
  std::cout << "Submitting AutoTask ...\n";
  runtime->submit(std::move(compound_auto_task));

  // Check the results
  std::cout << "\nFinal results:\n";

  // Display output values (computed by PhysicalTask within AutoTask)
  const legate::PhysicalStore output_p_store = output_store.get_physical_store();
  const auto output_shape                    = output_p_store.shape<1>();
  const auto output_span = output_p_store.span_read_accessor<std::int64_t, 1>(output_shape);
  std::cout << "Output values (after transformation): ";
  for (std::size_t i = 0; i < output_span.extent(0); ++i) {
    std::cout << output_span[i] << " ";
  }
  std::cout << "\n";

  std::cout << "AutoTask -> PhysicalTask demonstration completed!\n";
}

}  // namespace

int main()
{
  legate::start();

  compound_autotask_with_physicaltasks();

  return legate::finish();
}
