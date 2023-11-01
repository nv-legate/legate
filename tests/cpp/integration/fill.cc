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

#include "legate.h"
#include "utilities/utilities.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <tuple>

namespace fill_test {

constexpr const char* library_name = "test_fill";

constexpr std::size_t SIZE = 10;

enum TaskIDs {
  CHECK_TASK       = 0,
  CHECK_SLICE_TASK = 3,
};

using FillTests = DefaultFixture;

class Whole : public DefaultFixture,
              public ::testing::WithParamInterface<std::tuple<bool, std::int32_t, std::size_t>> {};

class Slice : public DefaultFixture,
              public ::testing::WithParamInterface<std::tuple<bool, std::int32_t>> {};

INSTANTIATE_TEST_SUITE_P(FillTests,
                         Whole,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(1, 2, 3),
                                            ::testing::Values(1, SIZE)));

INSTANTIATE_TEST_SUITE_P(FillTests,
                         Slice,
                         ::testing::Combine(::testing::Bool(), ::testing::Values(1, 2, 3)));

template <std::int32_t DIM>
struct CheckTask : public legate::LegateTask<CheckTask<DIM>> {
  static const std::int32_t TASK_ID = CHECK_TASK + DIM;
  static void cpu_variant(legate::TaskContext context);
};

template <std::int32_t DIM>
struct CheckSliceTask : public legate::LegateTask<CheckSliceTask<DIM>> {
  static const std::int32_t TASK_ID = CHECK_SLICE_TASK + DIM;
  static void cpu_variant(legate::TaskContext context);
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  CheckTask<1>::register_variants(context);
  CheckTask<2>::register_variants(context);
  CheckTask<3>::register_variants(context);
  CheckSliceTask<1>::register_variants(context);
  CheckSliceTask<2>::register_variants(context);
  CheckSliceTask<3>::register_variants(context);
}

template <std::int32_t DIM>
/*static*/ void CheckTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input    = context.input(0);
  auto shape    = input.shape<DIM>();
  int64_t value = context.scalar(0).value<int64_t>();

  if (shape.empty()) return;

  auto val_acc = input.data().read_accessor<int64_t, DIM>(shape);
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(val_acc[*it], value);
  }

  if (!input.nullable()) return;

  auto mask_acc = input.null_mask().read_accessor<bool, DIM>(shape);
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(mask_acc[*it], true);
  }
}

template <std::int32_t DIM>
/*static*/ void CheckSliceTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input               = context.input(0);
  auto shape               = input.shape<DIM>();
  auto value_in_slice      = context.scalar(0);
  auto value_outside_slice = context.scalar(1);
  auto offset              = context.scalar(2).value<int64_t>();

  if (shape.empty()) return;

  auto in_slice = [&offset](const auto& p) {
    for (std::int32_t dim = 0; dim < DIM; ++dim)
      if (p[dim] < offset) return false;
    return true;
  };

  if (!input.nullable()) {
    auto acc         = input.data().read_accessor<int64_t, DIM>(shape);
    auto v_in_slice  = value_in_slice.value<int64_t>();
    auto v_out_slice = value_outside_slice.value<int64_t>();
    for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
      EXPECT_EQ(acc[*it], in_slice(*it) ? v_in_slice : v_out_slice);
    }
    return;
  }

  auto val_acc    = input.data().read_accessor<int64_t, DIM>(shape);
  auto mask_acc   = input.null_mask().read_accessor<bool, DIM>(shape);
  auto v_in_slice = value_in_slice.value<int64_t>();
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    if (in_slice(*it)) {
      EXPECT_EQ(val_acc[*it], v_in_slice);
      EXPECT_EQ(mask_acc[*it], true);
    } else {
      EXPECT_EQ(mask_acc[*it], false);
    }
  }
}

void check_output(const legate::LogicalArray& array, const legate::Scalar& value)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto task = runtime->create_task(context, CHECK_TASK + array.dim());
  task.add_input(array);
  task.add_scalar_arg(value);
  runtime->submit(std::move(task));
}

void check_output_slice(const legate::LogicalArray& array,
                        const legate::Scalar& value_in_slice,
                        const legate::Scalar& value_outside_slice,
                        int64_t offset)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto task = runtime->create_task(context, CHECK_SLICE_TASK + array.dim());
  task.add_input(array);
  task.add_scalar_arg(value_in_slice);
  task.add_scalar_arg(value_outside_slice);
  task.add_scalar_arg(legate::Scalar{offset});
  runtime->submit(std::move(task));
}

void test_fill_index(std::int32_t dim, std::size_t size, bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();

  auto lhs = runtime->create_array(legate::full(static_cast<std::size_t>(dim), size),
                                   legate::int64(),
                                   nullable /*nullable*/,
                                   true /*optimize_scalar*/);
  auto v   = legate::Scalar{int64_t{10}};

  // fill input array with some values
  runtime->issue_fill(lhs, v);

  // check the result of fill
  check_output(lhs, v);
}

void test_fill_slice(std::int32_t dim, std::size_t size, bool null_init)
{
  auto runtime = legate::Runtime::get_runtime();

  constexpr int64_t v1     = 100;
  constexpr int64_t v2     = 200;
  constexpr int64_t offset = 3;

  auto lhs = runtime->create_array(
    legate::full(static_cast<std::size_t>(dim), size), legate::int64(), null_init);
  auto value_in_slice      = legate::Scalar{v1};
  auto value_outside_slice = null_init ? legate::Scalar{} : legate::Scalar{v2};

  // First fill the entire store with v2
  runtime->issue_fill(lhs, value_outside_slice);

  // Then fill a slice with v1
  auto sliced = lhs;
  for (std::int32_t idx = 0; idx < dim; ++idx) sliced = sliced.slice(idx, legate::Slice{offset});
  runtime->issue_fill(sliced, value_in_slice);

  // check if the slice is filled correctly
  check_output_slice(lhs, legate::Scalar{v1}, legate::Scalar{v2}, offset);
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10, 10}, legate::int64(), false /*nullable*/);
  auto v       = legate::Scalar{10.0};

  // Type mismatch
  EXPECT_THROW(runtime->issue_fill(array, runtime->create_store(v)), std::invalid_argument);
  EXPECT_THROW(runtime->issue_fill(array, v), std::invalid_argument);

  // Nulliyfing a non-nullable array
  EXPECT_THROW(runtime->issue_fill(array, legate::Scalar{}), std::invalid_argument);
}

TEST_P(Whole, Index)
{
  register_tasks();
  const auto& [nullable, dim, size] = GetParam();
  test_fill_index(dim, size, nullable);
}

TEST_P(Whole, Single)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  legate::MachineTracker tracker(machine.slice(0, 1, legate::mapping::TaskTarget::CPU));

  register_tasks();
  const auto& [nullable, dim, size] = GetParam();
  test_fill_index(dim, size, nullable);
}

TEST_P(Slice, Index)
{
  register_tasks();
  for (bool null_init : {false, true}) {
    for (std::int32_t dim : {1, 2, 3}) { test_fill_slice(dim, SIZE, null_init); }
  }
}

TEST_F(FillTests, Invalid) { test_invalid(); }

}  // namespace fill_test
