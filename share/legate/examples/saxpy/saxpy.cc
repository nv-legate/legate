/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <cstdint>
#include <iostream>
#include <string_view>

namespace {

// We define the tasks here, but at this point, Legate is not yet aware of it.
// The task cannot be used until it is registered with a library,
// after which Legate can recognize it and allow instances to be
// created and executed.

// saxpy task
class SAXPYTask : public legate::LegateTask<SAXPYTask> {
 public:
  //  This `TaskConfig` signature below does the following:
  //    - Assigns a local task ID of 1 ((should be unique within the library).
  //    - Specifies that the task takes exactly 2 input stores and 1 output
  //      store; this calling convention will be checked at runtime when
  //      the task is launched.
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(
      legate::TaskSignature{}.inputs(2).outputs(1));

  // This is the CPU variant of the task
  // there could be also OpenMP and CUDA variants in Legate
  static void cpu_variant(legate::TaskContext context)
  {
    // DIM is a dimension of the input/output arrays in this
    // example. We fix it to 2 as we are operating on 2D arrays here.
    // The same task can be reused for arrays different dimensions
    // if we template it on DIM
    constexpr std::int32_t DIM = 2;

    // Retrieve the input and output store handles from the task context
    const auto x_input_store = context.input(0).data();
    const auto y_input_store = context.input(1).data();
    auto output_store        = context.output(0).data();

    // Extract shapes
    const auto x_shape      = x_input_store.shape<DIM>();
    const auto y_shape      = y_input_store.shape<DIM>();
    const auto output_shape = output_store.shape<DIM>();

    // Assert that all shapes are equal
    LEGATE_ASSERT(x_shape == y_shape && y_shape == output_shape);

    // Create accessors for reading input and writing output
    auto x_accessor      = x_input_store.read_accessor<std::int32_t, DIM>();
    auto y_accessor      = y_input_store.read_accessor<std::int32_t, DIM>();
    auto output_accessor = output_store.write_accessor<std::int32_t, DIM>();

    // Iterate over each point in the 2D domain (dense rectangular iteration)
    for (legate::PointInRectIterator<DIM> it{x_shape}; it.valid(); ++it) {
      // Perform the element-wise SAXPY operation: output = x + y
      output_accessor[*it] = x_accessor[*it] + y_accessor[*it];
    }
  }
};

// verification task
class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    constexpr std::int32_t DIM = 2;
    const auto result          = context.input(0).data();
    const auto expected        = context.scalar(0).value<std::int32_t>();
    const auto shape           = result.shape<DIM>();
    auto acc                   = result.read_accessor<std::int32_t, DIM>();

    bool all_correct = true;
    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      if (acc[*it] != expected) {
        std::cerr << "Error at position (" << (*it)[0] << "," << (*it)[1] << "): expected "
                  << expected << ", got " << acc[*it] << "\n";
        all_correct = false;
      }
    }

    if (all_correct) {
      std::cout << "All " << shape.volume() << " values correct in 2D array!\n";
    }
  }
};

void run_example()
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->create_library("legate_saxpy");
  // In a Legate application, tasks must be registered with
  // their corresponding library before any instances can be created.
  // Since an application may use several Legate-based libraries,
  // it's important to separate their task registration logic.
  SAXPYTask::register_variants(library);
  CheckTask::register_variants(library);

  // Create 2D arrays (5x5)
  auto shape   = legate::Shape{5, 5};
  auto x_array = runtime->create_array(shape, legate::int32());
  auto y_array = runtime->create_array(shape, legate::int32());
  auto result  = runtime->create_array(shape, legate::int32());

  // Fill input arrays
  runtime->issue_fill(x_array, legate::Scalar{1});  // Fill with 1s
  runtime->issue_fill(y_array, legate::Scalar{2});  // Fill with 2s

  // Create and submit saxpy task
  auto saxpy_task = runtime->create_task(library, SAXPYTask::TASK_CONFIG.task_id());
  saxpy_task.add_input(x_array);
  saxpy_task.add_input(y_array);
  saxpy_task.add_output(result);
  // When runtime->submit is called, the task is enqueued
  // into the runtime's task execution queue. Legate will
  // analyze data dependencies between all submitted tasks and
  // schedule them asynchronously whenever possible.
  runtime->submit(std::move(saxpy_task));

  // Since check_task receives result as one of its inputs,
  // it has a data dependency on the output of saxpy_task.
  // Therefore, check_task will automatically be delayed until
  // saxpy_task finishes execution and produces the required data.

  // Create and submit verification task
  auto check_task = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());
  check_task.add_input(result);
  check_task.add_scalar_arg(legate::Scalar{3});  // Expected sum
  runtime->submit(std::move(check_task));
}

}  // namespace

int main()
{
  legate::start();
  run_example();
  return legate::finish();
}
