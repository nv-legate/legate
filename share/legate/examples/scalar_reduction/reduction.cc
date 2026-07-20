/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <reduction.h>

#include <legate.h>

#include <legate/redop/base.h>

#include <cstdint>
#include <iostream>
#include <string_view>

void InitTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 1;

  auto store       = context.output(0);
  const auto value = context.scalar(0).value<std::uint64_t>();
  const auto shape = store.shape<DIM>();

  auto accessor = store.write_accessor<std::uint64_t, DIM>();

  for (auto&& point : shape.iterate()) {
    accessor[point] = value;
  }
}

void ReductionTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 1;

  const auto input = context.input(0);
  auto output      = context.reduction(0);

  const auto shape = input.shape<DIM>();

  auto in_acc  = input.read_accessor<std::uint64_t, DIM>();
  auto red_acc = output.reduce_accessor<legate::SumReduction<std::uint64_t>, false, DIM>();

  for (auto&& point : shape.iterate()) {
    red_acc.reduce(0, in_acc[point]);
  }
}

void run_example()
{
  constexpr std::int32_t DIM = 1;

  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->create_library("reduction");
  InitTask::register_variants(library);
  ReductionTask::register_variants(library);

  auto count       = 2ULL << 24;
  auto shape       = legate::Shape{count};
  auto input_store = runtime->create_store(shape, legate::uint64());

  auto init_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  init_task.add_output(input_store);
  init_task.add_scalar_arg(legate::Scalar{std::uint64_t{1}});
  runtime->submit(std::move(init_task));

  auto init         = legate::Scalar{std::uint64_t{0}};
  auto reduce_store = runtime->create_store(init);

  auto reduce_task = runtime->create_task(library, ReductionTask::TASK_CONFIG.task_id());
  reduce_task.add_input(input_store);
  reduce_task.add_reduction(reduce_store, legate::ReductionOpKind::ADD);
  runtime->submit(std::move(reduce_task));

  auto p_reduce_store   = reduce_store.get_physical_store(legate::mapping::StoreTarget::SYSMEM);
  auto acc_reduce_store = p_reduce_store.read_accessor<std::uint64_t, DIM>();

  if (acc_reduce_store[0] != static_cast<std::uint64_t>(count) + init.value<std::uint64_t>()) {
    std::cerr << "Test failed: expected " << (count + init.value<std::uint64_t>()) << ", got "
              << acc_reduce_store[0] << "\n";
  } else {
    std::cerr << "Test passed" << "\n";
  }
}

int main()
{
  legate::start();
  run_example();
  return legate::finish();
}
