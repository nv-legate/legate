/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "core/mapping/mapping.h"
#include "legate.h"

namespace multiscalarout {

static const char* library_name = "multi_scalar";
static legate::Logger logger(library_name);

enum TaskIDs {
  WRITER  = 0,
  REDUCER = 1,
};

struct WriterTask : public legate::LegateTask<WriterTask> {
  static const int32_t TASK_ID = WRITER;
  static void cpu_variant(legate::TaskContext& context);
};

struct ReducerTask : public legate::LegateTask<ReducerTask> {
  static const int32_t TASK_ID = REDUCER;
  static void cpu_variant(legate::TaskContext& context);
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  WriterTask::register_variants(context);
  ReducerTask::register_variants(context);
}

/*static*/ void WriterTask::cpu_variant(legate::TaskContext& context)
{
  auto& output1 = context.outputs()[0];
  auto& output2 = context.outputs()[1];

  auto acc1 = output1.write_accessor<int8_t, 2>();
  auto acc2 = output2.write_accessor<int32_t, 3>();

  acc1[{0, 0}]    = 10;
  acc2[{0, 0, 0}] = 20;
}

/*static*/ void ReducerTask::cpu_variant(legate::TaskContext& context)
{
  auto& red1 = context.reductions()[0];
  auto& red2 = context.reductions()[1];

  auto acc1 = red1.reduce_accessor<legate::SumReduction<int8_t>, true, 2>();
  auto acc2 = red2.reduce_accessor<legate::ProdReduction<int32_t>, true, 3>();

  acc1[{0, 0}].reduce(10);
  acc2[{0, 0, 0}].reduce(2);
}

void test_writer_auto(legate::LibraryContext* context,
                      legate::LogicalStore scalar1,
                      legate::LogicalStore scalar2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, WRITER);
  auto part1   = task->declare_partition();
  auto part2   = task->declare_partition();
  task->add_output(scalar1, part1);
  task->add_output(scalar2, part2);
  runtime->submit(std::move(task));
}

void test_reducer_auto(legate::LibraryContext* context,
                       legate::LogicalStore scalar1,
                       legate::LogicalStore scalar2,
                       legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, REDUCER);
  auto part1   = task->declare_partition();
  auto part2   = task->declare_partition();
  auto part3   = task->declare_partition();
  auto redop1  = scalar1.type().find_reduction_operator(legate::ReductionOpKind::ADD);
  auto redop2  = scalar1.type().find_reduction_operator(legate::ReductionOpKind::MUL);
  task->add_reduction(scalar1, redop1, part1);
  task->add_reduction(scalar2, redop2, part2);
  task->add_output(store, part3);
  runtime->submit(std::move(task));
}

void test_reducer_manual(legate::LibraryContext* context,
                         legate::LogicalStore scalar1,
                         legate::LogicalStore scalar2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, REDUCER, legate::Shape({2}));
  auto redop1  = scalar1.type().find_reduction_operator(legate::ReductionOpKind::ADD);
  auto redop2  = scalar2.type().find_reduction_operator(legate::ReductionOpKind::MUL);
  task->add_reduction(scalar1, redop1);
  task->add_reduction(scalar2, redop2);
  runtime->submit(std::move(task));
}

void print_stores(legate::LibraryContext* context,
                  legate::LogicalStore scalar1,
                  legate::LogicalStore scalar2)
{
  auto runtime   = legate::Runtime::get_runtime();
  auto p_scalar1 = scalar1.get_physical_store(context);
  auto p_scalar2 = scalar2.get_physical_store(context);
  auto acc1      = p_scalar1->read_accessor<int8_t, 2>();
  auto acc2      = p_scalar2->read_accessor<int32_t, 3>();
  std::stringstream ss;
  ss << static_cast<int32_t>(acc1[{0, 0}]) << " " << acc2[{0, 0, 0}];
  logger.print() << ss.str();
}

TEST(Integration, ManualScalarOut)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto scalar1 = runtime->create_store({1, 1}, legate::int8(), true);
  auto scalar2 = runtime->create_store({1, 1, 1}, legate::int32(), true);
  auto store   = runtime->create_store({10}, legate::int64());
  test_writer_auto(context, scalar1, scalar2);
  print_stores(context, scalar1, scalar2);
  test_reducer_auto(context, scalar1, scalar2, store);
  print_stores(context, scalar1, scalar2);
  test_reducer_manual(context, scalar1, scalar2);
  print_stores(context, scalar1, scalar2);
}

}  // namespace multiscalarout
