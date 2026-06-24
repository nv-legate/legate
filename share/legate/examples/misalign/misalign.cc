/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <misalign.h>

#include <legate.h>

#include <cstdint>
#include <iostream>
#include <string_view>

void InitTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 3;

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
    if (acc[*it] != expected + (*it)[0] * 2 + offset) {
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
  auto library        = runtime->create_library("misalign");
  InitTask::register_variants(library);
  DummyTask::register_variants(library);
  AddTask::register_variants(library);
  CheckTask::register_variants(library);

  const auto x_shape = legate::Shape{21, 40, 20};
  const auto y_shape = legate::Shape{10, 40};

  auto x_store = runtime->create_store(x_shape, legate::int32());
  auto y_store = runtime->create_store(y_shape, legate::int32());

  auto init_x_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  init_x_task.add_output(x_store);
  init_x_task.add_scalar_arg(legate::Scalar{1});
  runtime->submit(std::move(init_x_task));

  auto dummy_task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
  dummy_task.add_input(x_store);
  runtime->submit(std::move(dummy_task));

  auto slice1 = x_store.slice(0, legate::Slice{11, legate::Slice::OPEN}).project(2, 1);
  auto slice2 = x_store.slice(0, legate::Slice{legate::Slice::OPEN, -11}).project(2, 1);

  auto add_task    = runtime->create_task(library, AddTask::TASK_CONFIG.task_id());
  auto add_s1_part = add_task.add_input(slice1);
  auto add_s2_part = add_task.add_input(slice2);
  auto add_y_part  = add_task.add_output(y_store);
  add_task.add_constraint(legate::align(add_s1_part, add_s2_part));
  add_task.add_constraint(legate::align(add_s1_part, add_y_part));
  runtime->submit(std::move(add_task));

  auto check_task = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());
  check_task.add_input(y_store);
  check_task.add_scalar_arg(legate::Scalar{2});   // Expected sum
  check_task.add_scalar_arg(legate::Scalar{11});  // Offset
  runtime->submit(std::move(check_task));
}

int main()
{
  legate::start();
  run_example();
  return legate::finish();
}
