/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <slice.h>

#include <legate.h>

#include <cstdint>
#include <iostream>
#include <string_view>

void InitTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;

  auto store       = context.output(0);
  const auto value = context.scalar(0).value<std::int32_t>();
  const auto shape = store.shape<DIM>();

  auto accessor = store.write_accessor<std::int32_t, DIM>();

  for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
    accessor[*it] = value + (*it)[0];
  }
}

void AddTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;

  const auto x_input_store = context.input(0);
  const auto y_input_store = context.input(1);
  auto output_store        = context.output(0);

  const auto x_shape      = x_input_store.shape<DIM>();
  const auto y_shape      = y_input_store.shape<DIM>();
  const auto output_shape = output_store.shape<DIM>();

  LEGATE_CHECK(x_shape == y_shape && y_shape == output_shape);

  auto x_accessor      = x_input_store.read_accessor<std::int32_t, DIM>();
  auto y_accessor      = y_input_store.read_accessor<std::int32_t, DIM>();
  auto output_accessor = output_store.write_accessor<std::int32_t, DIM>();

  for (legate::PointInRectIterator<DIM> it{x_shape}; it.valid(); ++it) {
    output_accessor[*it] = x_accessor[*it] + y_accessor[*it];
  }
}

void CopyTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;
  const auto input           = context.input(0);
  const auto output          = context.output(0);
  const auto shape           = input.shape<DIM>();
  const auto in_acc          = input.read_accessor<std::int32_t, DIM>();
  auto out_acc               = output.write_accessor<std::int32_t, DIM>();

  for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
    out_acc[*it] = in_acc[*it];
  }
}

void CheckTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;
  const auto result          = context.input(0);
  const auto expected        = context.scalar(0).value<std::int32_t>();
  const auto offset          = context.scalar(1).value<std::int32_t>();
  const auto shape           = result.shape<DIM>();
  auto acc                   = result.read_accessor<std::int32_t, DIM>();

  std::cerr << shape << std::endl;
  bool all_correct = true;
  for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
    if (acc[*it] != expected + ((*it)[0] + offset) * 2) {
      std::cerr << "Error at position (" << (*it)[0] << "," << (*it)[1] << "): expected "
                << expected << ", got " << acc[*it] << "\n";
      all_correct = false;
    }
  }

  if (all_correct) {
    std::cout << "Task " << context.get_task_index() << ": All " << shape.volume()
              << " values correct in 2D array!\n";
  }
}

void run_example()
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->create_library("legate_saxpy");
  // In a Legate application, tasks must be registered with
  // their corresponding library before any instances can be created.
  // Since an application may use several Legate-based libraries,
  // it's important to separate their task registration logic.
  InitTask::register_variants(library);
  AddTask::register_variants(library);
  CopyTask::register_variants(library);
  CheckTask::register_variants(library);

  // Create 2D arrays
  auto shape   = legate::Shape{2048, 2048};
  auto x_store = runtime->create_store(shape, legate::int32());
  auto y_store = runtime->create_store(shape, legate::int32());
  auto result  = runtime->create_store(shape, legate::int32());

  // Fill input arrays
  auto init_x_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  init_x_task.add_output(x_store);
  init_x_task.add_scalar_arg(legate::Scalar{1});
  runtime->submit(std::move(init_x_task));

  auto init_y_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  init_y_task.add_output(y_store);
  init_y_task.add_scalar_arg(legate::Scalar{2});
  runtime->submit(std::move(init_y_task));

  // Create and submit saxpy task
  auto saxpy_task     = runtime->create_task(library, AddTask::TASK_CONFIG.task_id());
  auto saxpy_x_part   = saxpy_task.add_input(x_store);
  auto saxpy_y_part   = saxpy_task.add_input(y_store);
  auto saxpy_res_part = saxpy_task.add_output(result);
  saxpy_task.add_constraint(legate::align(saxpy_x_part, saxpy_y_part));
  saxpy_task.add_constraint(legate::align(saxpy_x_part, saxpy_res_part));
  // When runtime->submit is called, the task is enqueued
  // into the runtime's task execution queue. Legate will
  // analyze data dependencies between all submitted tasks and
  // schedule them asynchronously whenever possible.
  runtime->submit(std::move(saxpy_task));

  auto slice = result.slice(0, legate::Slice{768, 1280}).slice(1, legate::Slice{768, 1280});
  auto copy  = runtime->create_store(slice.shape(), legate::int32());

  auto copy_task     = runtime->create_task(library, CopyTask::TASK_CONFIG.task_id());
  auto copy_in_part  = copy_task.add_input(slice);
  auto copy_out_part = copy_task.add_output(copy);
  copy_task.add_constraint(legate::align(copy_in_part, copy_out_part));
  runtime->submit(std::move(copy_task));

  // Since check_task receives result as one of its inputs,
  // it has a data dependency on the output of saxpy_task.
  // Therefore, check_task will automatically be delayed until
  // saxpy_task finishes execution and produces the required data.

  // Create and submit verification task
  auto check_task = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());
  check_task.add_input(copy);
  check_task.add_scalar_arg(legate::Scalar{3});    // Expected sum
  check_task.add_scalar_arg(legate::Scalar{768});  // Offset
  runtime->submit(std::move(check_task));
}

int main()
{
  legate::start();
  run_example();
  return legate::finish();
}
