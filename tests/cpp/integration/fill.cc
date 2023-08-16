/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gtest/gtest.h>

#include "legate.h"

namespace fill {

static const char* library_name = "test_fill";

constexpr size_t SIZE = 10;

enum TaskIDs {
  CHECK_TASK       = 0,
  CHECK_SLICE_TASK = 3,
};

template <int32_t DIM>
struct CheckTask : public legate::LegateTask<CheckTask<DIM>> {
  static const int32_t TASK_ID = CHECK_TASK + DIM;
  static void cpu_variant(legate::TaskContext context);
};

template <int32_t DIM>
struct CheckSliceTask : public legate::LegateTask<CheckSliceTask<DIM>> {
  static const int32_t TASK_ID = CHECK_SLICE_TASK + DIM;
  static void cpu_variant(legate::TaskContext context);
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  CheckTask<1>::register_variants(context);
  CheckTask<2>::register_variants(context);
  CheckTask<3>::register_variants(context);
  CheckSliceTask<1>::register_variants(context);
  CheckSliceTask<2>::register_variants(context);
  CheckSliceTask<3>::register_variants(context);
}

template <int32_t DIM>
/*static*/ void CheckTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input    = context.input(0).data();
  auto shape    = input.shape<DIM>();
  int64_t value = context.scalar(0).value<int64_t>();

  if (shape.empty()) return;

  auto acc = input.write_accessor<int64_t, DIM>(shape);
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) { EXPECT_EQ(acc[*it], value); }
}

template <int32_t DIM>
/*static*/ void CheckSliceTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input                  = context.input(0).data();
  auto shape                  = input.shape<DIM>();
  int64_t value_in_slice      = context.scalar(0).value<int64_t>();
  int64_t value_outside_slice = context.scalar(1).value<int64_t>();
  int64_t offset              = context.scalar(2).value<int64_t>();

  if (shape.empty()) return;

  auto acc      = input.write_accessor<int64_t, DIM>(shape);
  auto in_slice = [&offset](const auto& p) {
    for (int32_t dim = 0; dim < DIM; ++dim)
      if (p[dim] < offset) return false;
    return true;
  };
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(acc[*it], in_slice(*it) ? value_in_slice : value_outside_slice);
  }
}

void check_output(legate::LogicalStore store, int64_t value)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto task = runtime->create_task(context, CHECK_TASK + store.dim());
  auto part = task.declare_partition();
  task.add_input(store, part);
  task.add_scalar_arg(value);
  runtime->submit(std::move(task));
}

void check_output_slice(legate::LogicalStore store,
                        int64_t value_in_slice,
                        int64_t value_outside_slice,
                        int64_t offset)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto task = runtime->create_task(context, CHECK_SLICE_TASK + store.dim());
  auto part = task.declare_partition();
  task.add_input(store, part);
  task.add_scalar_arg(value_in_slice);
  task.add_scalar_arg(value_outside_slice);
  task.add_scalar_arg(offset);
  runtime->submit(std::move(task));
}

void test_fill_index(int32_t dim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  int64_t v = 10;

  auto lhs   = runtime->create_store(legate::Shape(dim, SIZE), legate::int64());
  auto value = runtime->create_store(v);

  // fill input store with some values
  runtime->issue_fill(lhs, value);

  // check the result of fill
  check_output(lhs, v);
}

void test_fill_single(int32_t dim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  legate::MachineTracker tracker(machine.slice(0, 1, legate::mapping::TaskTarget::CPU));
  test_fill_index(dim);
}

void test_fill_slice(int32_t dim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  int64_t v1     = 100;
  int64_t v2     = 200;
  int64_t offset = 3;

  auto lhs                 = runtime->create_store(legate::Shape(dim, SIZE), legate::int64());
  auto value_in_slice      = runtime->create_store(v1);
  auto value_outside_slice = runtime->create_store(v2);

  // First fill the entire store with v1
  runtime->issue_fill(lhs, value_outside_slice);

  // Then fill a slice with v2
  auto sliced = lhs;
  for (int32_t idx = 0; idx < dim; ++idx) sliced = sliced.slice(idx, legate::Slice(offset));
  runtime->issue_fill(sliced, value_in_slice);

  // check if the slice is filled correctly
  check_output_slice(lhs, v1, v2, offset);
}

TEST(Integration, FillIndex)
{
  legate::Core::perform_registration<register_tasks>();
  test_fill_index(1);
  test_fill_index(2);
  test_fill_index(3);
}

TEST(Integration, FillSingle)
{
  legate::Core::perform_registration<register_tasks>();
  test_fill_single(1);
  test_fill_single(2);
  test_fill_single(3);
}

TEST(Integration, FillSlice)
{
  legate::Core::perform_registration<register_tasks>();
  test_fill_slice(1);
  test_fill_slice(2);
  test_fill_slice(3);
}

}  // namespace fill
