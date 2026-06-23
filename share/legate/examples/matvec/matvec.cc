/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <matvec.h>

#include <legate.h>

#include <legate/redop/base.h>

#include <cstdint>
#include <iostream>
#include <string_view>

template <std::int32_t DIM>
void InitTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto store       = context.output(0);
  const auto value = context.scalar(0).value<std::int32_t>();
  const auto shape = store.shape<DIM>();

  auto accessor = store.write_accessor<std::int32_t, DIM>();

  for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
    accessor[*it] = value;
  }
}

void MatVecTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;
  const auto rhs1            = context.input(0);
  const auto rhs2            = context.input(1);
  const auto lhs             = context.reduction(0);
  const auto shape           = rhs1.shape<DIM>();
  const auto rhs1_acc        = rhs1.read_accessor<std::int32_t, DIM>();
  const auto rhs2_acc        = rhs2.read_accessor<std::int32_t, DIM>();
  auto lhs_acc = lhs.reduce_accessor<legate::SumReduction<std::int32_t>, /*exclusive=*/true, DIM>();

  for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
    lhs_acc.reduce(*it, rhs1_acc[*it] * rhs2_acc[*it]);
  }
}

void CheckTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 1;
  const auto result          = context.input(0);
  const auto expected        = static_cast<std::int32_t>(context.scalar(0).value<std::uint64_t>());
  const auto shape           = result.shape<DIM>();
  const auto acc             = result.read_accessor<std::int32_t, DIM>();

  bool all_correct = true;
  for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
    if (acc[*it] != expected) {
      std::cerr << "Error at position " << (*it)[0] << ": expected " << expected << ", got "
                << acc[*it] << "\n";
      all_correct = false;
    }
  }

  if (all_correct) {
    std::cout << "Task " << context.get_task_index() << ": All " << shape.volume()
              << " values correct!\n";
  }
}

void run_example()
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->create_library("matvec");

  InitTask<1>::register_variants(library);
  InitTask<2>::register_variants(library);
  DummyTask::register_variants(library);
  MatVecTask::register_variants(library);
  CheckTask::register_variants(library);

  constexpr std::uint64_t OUT_DIM = 1000;
  constexpr std::uint64_t RED_DIM = 1200;

  auto rhs1 = runtime->create_store(legate::Shape{OUT_DIM, RED_DIM}, legate::int32());
  auto rhs2 = runtime->create_store(legate::Shape{RED_DIM}, legate::int32());
  auto lhs  = runtime->create_store(legate::Shape{OUT_DIM}, legate::int32());

  auto init_rhs1_task = runtime->create_task(library, InitTask<2>::TASK_CONFIG.task_id());
  init_rhs1_task.add_output(rhs1);
  init_rhs1_task.add_scalar_arg(legate::Scalar{1});
  runtime->submit(std::move(init_rhs1_task));

  auto init_rhs2_task = runtime->create_task(library, InitTask<1>::TASK_CONFIG.task_id());
  init_rhs2_task.add_output(rhs2);
  init_rhs2_task.add_scalar_arg(legate::Scalar{1});
  runtime->submit(std::move(init_rhs2_task));

  auto init_lhs_task = runtime->create_task(library, InitTask<1>::TASK_CONFIG.task_id());
  init_lhs_task.add_output(lhs);
  init_lhs_task.add_scalar_arg(legate::Scalar{0});
  runtime->submit(std::move(init_lhs_task));

  auto dummy_task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
  dummy_task.add_input(lhs);
  runtime->submit(std::move(dummy_task));

  auto matvec_task = runtime->create_task(library, MatVecTask::TASK_CONFIG.task_id());
  auto rhs1_part   = matvec_task.add_input(rhs1);
  auto rhs2_part   = matvec_task.add_input(rhs2.promote(0, OUT_DIM));
  auto lhs_part = matvec_task.add_reduction(lhs.promote(1, RED_DIM), legate::ReductionOpKind::ADD);
  matvec_task.add_constraint(legate::align(rhs1_part, rhs2_part));
  matvec_task.add_constraint(legate::align(lhs_part, rhs1_part));
  runtime->submit(std::move(matvec_task));

  auto check_task = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());
  check_task.add_input(lhs);
  check_task.add_scalar_arg(legate::Scalar{RED_DIM});
  runtime->submit(std::move(check_task));
}

int main()
{
  legate::start();
  run_example();
  return legate::finish();
}
