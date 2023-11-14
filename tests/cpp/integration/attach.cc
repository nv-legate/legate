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

#include "core/data/detail/logical_store.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace attach {

using Attach          = DefaultFixture;
using AttachDeathTest = DeathTestFixture;

static const char* library_name = "test_attach";

enum TaskOpCode {
  ADDER   = 0,
  CHECKER = 1,
};

static legate::Shape SHAPE_1D{5};

static legate::Shape SHAPE_2D{3, 4};

void increment_physical_store(const legate::PhysicalStore& store, int32_t dim)
{
  if (dim == 1) {
    auto shape = store.shape<1>();
    auto acc   = store.read_write_accessor<int64_t, 1, true>(shape);
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) acc[*it] += 1;
  } else {
    auto shape = store.shape<2>();
    auto acc   = store.read_write_accessor<int64_t, 2, true>(shape);
    for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it) acc[*it] += 1;
  }
}

void check_physical_store(const legate::PhysicalStore& store, int32_t dim, int64_t counter)
{
  if (dim == 1) {
    auto shape = store.shape<1>();
    auto acc   = store.read_accessor<int64_t, 1, true>(shape);
    for (size_t i = 0; i < SHAPE_1D[0]; ++i) EXPECT_EQ(acc[i], counter++);
  } else {
    auto shape = store.shape<2>();
    auto acc   = store.read_accessor<int64_t, 2, true>(shape);
    // Legate should always see elements in the expected order
    for (size_t i = 0; i < SHAPE_2D[0]; ++i)
      for (size_t j = 0; j < SHAPE_2D[1]; ++j) EXPECT_EQ(acc[legate::Point<2>(i, j)], counter++);
  }
}

struct AdderTask : public legate::LegateTask<AdderTask> {
  static const int32_t TASK_ID = ADDER;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    int32_t dim = context.scalar(0).value<int32_t>();
    increment_physical_store(output, dim);
  }
};

struct CheckerTask : public legate::LegateTask<CheckerTask> {
  static const int32_t TASK_ID = CHECKER;
  static void cpu_variant(legate::TaskContext context)
  {
    auto input      = context.input(0).data();
    int32_t dim     = context.scalar(0).value<int32_t>();
    int64_t counter = context.scalar(1).value<int64_t>();
    check_physical_store(input, dim, counter);
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  AdderTask::register_variants(context);
  CheckerTask::register_variants(context);
}

int64_t* make_buffer(int32_t dim, bool fortran)
{
  int64_t* buffer;
  int64_t counter = 0;
  if (dim == 1) {
    buffer = new int64_t[SHAPE_1D.volume()];
    for (size_t i = 0; i < SHAPE_1D[0]; ++i) buffer[i] = counter++;
  } else {
    buffer = new int64_t[SHAPE_2D.volume()];
    for (size_t i = 0; i < SHAPE_2D[0]; ++i)
      for (size_t j = 0; j < SHAPE_2D[1]; ++j)
        if (fortran)
          buffer[j * SHAPE_2D[0] + i] = counter++;
        else
          buffer[i * SHAPE_2D[1] + j] = counter++;
  }
  return buffer;
}

void check_and_delete_buffer(int64_t* buffer, int32_t dim, bool fortran, int64_t counter)
{
  if (dim == 1)
    for (size_t i = 0; i < SHAPE_1D[0]; ++i) EXPECT_EQ(buffer[i], counter++);
  else
    for (size_t i = 0; i < SHAPE_2D[0]; ++i)
      for (size_t j = 0; j < SHAPE_2D[1]; ++j)
        if (fortran)
          EXPECT_EQ(buffer[j * SHAPE_2D[0] + i], counter++);
        else
          EXPECT_EQ(buffer[i * SHAPE_2D[1] + j], counter++);
  delete[] buffer;
}

void test_body(
  int32_t dim, bool fortran, bool unordered, bool share, bool use_tasks, bool use_inline)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto context    = runtime->find_library(library_name);
  int64_t counter = 0;
  auto buffer     = make_buffer(dim, fortran);
  auto l_store    = runtime->create_store(dim == 1 ? SHAPE_1D : SHAPE_2D,
                                       legate::int64(),
                                       buffer,
                                       share,
                                       fortran ? legate::mapping::DimOrdering::fortran_order()
                                               : legate::mapping::DimOrdering::c_order());
  if (unordered) l_store.impl()->allow_out_of_order_destruction();
  if (!share) check_and_delete_buffer(buffer, dim, fortran, counter);
  for (auto iter = 0; iter < 2; ++iter) {
    if (use_tasks) {
      auto task = runtime->create_task(context, ADDER, legate::Shape{1});
      task.add_input(l_store);
      task.add_output(l_store);
      task.add_scalar_arg(legate::Scalar{dim});
      runtime->submit(std::move(task));
      ++counter;
    }
    if (use_inline) {
      auto p_store = l_store.get_physical_store();
      increment_physical_store(p_store, dim);
      ++counter;
    }
  }
  if (use_tasks) {
    auto task = runtime->create_task(context, CHECKER, legate::Shape{1});
    task.add_input(l_store);
    task.add_scalar_arg(legate::Scalar{dim});
    task.add_scalar_arg(legate::Scalar{counter});
    runtime->submit(std::move(task));
  }
  if (use_inline) {
    auto p_store = l_store.get_physical_store();
    check_physical_store(p_store, dim, counter);
  }
  if (share) {
    l_store.detach();
    check_and_delete_buffer(buffer, dim, fortran, counter);
  }
}

TEST_F(Attach, Positive)
{
  register_tasks();
  // Do multiple operations in a row, with stores collected in-between, in hopes of triggering
  // consensus match.
  // TODO: Also try keeping multiple stores alive at one time.
  for (auto& [dim, fortran] : std::vector<std::pair<int32_t, bool>>{
         std::make_pair(1, false), std::make_pair(2, false), std::make_pair(2, true)})
    for (bool unordered : {false, true})
      for (bool share : {false, true})
        for (bool use_tasks : {false, true})
          for (bool use_inline : {false, true})
            test_body(dim, fortran, unordered, share, use_tasks, use_inline);
}

TEST_F(Attach, Negative)
{
  register_tasks();
  auto runtime = legate::Runtime::get_runtime();

  // Trying to detach a store without an attachment
  EXPECT_THROW(runtime->create_store(legate::Scalar(42)).detach(), std::invalid_argument);
  EXPECT_THROW(runtime->create_store(SHAPE_2D, legate::int64()).detach(), std::invalid_argument);
  EXPECT_THROW(runtime->create_store(legate::int64()).detach(), std::invalid_argument);

  // Trying to attach to a NULL buffer
  EXPECT_THROW((void)runtime->create_store(SHAPE_2D, legate::int64(), nullptr, true),
               std::invalid_argument);

  {
    auto mem = new int64_t[SHAPE_1D.volume()];
    // Trying to detach a non-shared attachment
    EXPECT_THROW(runtime->create_store(SHAPE_1D, legate::int64(), mem).detach(),
                 std::invalid_argument);
    delete[] mem;
  }

  {
    // Trying to detach a sub-store
    auto mem     = new int64_t[SHAPE_1D.volume()];
    auto l_store = runtime->create_store(SHAPE_1D, legate::int64(), mem, true /*share*/);
    EXPECT_THROW(l_store.project(0, 1).detach(), std::invalid_argument);
    // We have to properly detach this, to avoid the abort in the destructor
    l_store.detach();
    delete[] mem;
  }
}

TEST_F(AttachDeathTest, MissingManualDetach)
{
  // Forgetting to detach a shared attachment
  // TODO: We can't check the actual error message, because it's reported using logging, and even
  // though that's mirrored on stderr, it doesn't seem to be included in the death message.
  EXPECT_DEATH(
    {
      (void)legate::start(argc_, argv_);
      register_tasks();
      auto runtime = legate::Runtime::get_runtime();
      (void)runtime->create_store(
        SHAPE_1D, legate::int64(), new int64_t[SHAPE_1D.volume()], true /*share*/);
      (void)legate::finish();
    },
    "");
}

}  // namespace attach
