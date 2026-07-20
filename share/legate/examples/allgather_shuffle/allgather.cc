/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <allgather.h>

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

  for (auto&& point : shape.iterate()) {
    accessor[point] = value + point[0];
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

  for (auto&& point : shape.iterate()) {
    out_acc[point] = in_acc[point];
  }
}

void CheckTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM                    = 2;
  constexpr std::uint32_t MAX_INCORRECT_RESULTS = 10;

  const auto result   = context.input(0);
  const auto expected = context.scalar(0).value<std::int32_t>();
  const auto shape    = result.shape<DIM>();
  auto acc            = result.read_accessor<std::int32_t, DIM>();

  std::cerr << shape << std::endl;
  bool all_correct              = true;
  std::uint32_t incorrect_count = 0;

  for (auto&& point : shape.iterate()) {
    if (acc[point] != expected + point[0]) {
      ++incorrect_count;
      std::cerr << "Error at position (" << point[0] << "," << point[1] << "): expected "
                << expected << ", got " << acc[point] << "\n";
      all_correct = false;
    }
    if (incorrect_count >= MAX_INCORRECT_RESULTS) {
      break;
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
  auto library        = runtime->create_library("allgather");

  InitTask::register_variants(library);
  DummyTask::register_variants(library);
  CopyTask::register_variants(library);
  CheckTask::register_variants(library);

  auto x_shape = legate::Shape{2048, 2048};
  auto y_shape = legate::Shape{2047, 2047};
  auto x_store = runtime->create_store(x_shape, legate::int32());
  auto y_store = runtime->create_store(y_shape, legate::int32());

  auto init_x_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  auto part        = init_x_task.add_output(x_store);
  init_x_task.add_scalar_arg(legate::Scalar{42});
  init_x_task.add_constraint(legate::broadcast(part, {1}));
  runtime->submit(std::move(init_x_task));

  // Right now we can't do a direct allgather or reshuffling from CPUs to GPUs
  // so we first launch a dummy task just to move the data to GPUs
  auto dummy_task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
  dummy_task.add_input(x_store);
  runtime->submit(std::move(dummy_task));

  auto x_slice   = x_store.slice(0, legate::Slice{1, legate::Slice::OPEN})
                     .slice(1, legate::Slice{1, legate::Slice::OPEN});
  auto copy_task = runtime->create_task(library, CopyTask::TASK_CONFIG.task_id());
  auto part_x    = copy_task.add_input(x_slice);
  auto part_y    = copy_task.add_output(y_store);
  copy_task.add_constraint(legate::broadcast(part_x));
  copy_task.add_constraint(legate::align(part_x, part_y));
  runtime->submit(std::move(copy_task));

  auto check_task = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());
  check_task.add_input(y_store);
  // x is initialized so that x[i][j] == 42 + i, and we copy x[1:, 1:] into y, so
  // y[i][j] == 42 + (i + 1) == 43 + i. Account for the row offset introduced by the slice.
  check_task.add_scalar_arg(legate::Scalar{43});
  runtime->submit(std::move(check_task));
}

int main()
{
  legate::start();
  run_example();
  return legate::finish();
}
