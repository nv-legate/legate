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

#include "legate.h"

namespace tree_reduce {

static const char* library_name = "test_tree_reduce";

static const size_t TILE_SIZE = 10;

enum TaskIDs {
  TASK_PRODUCE_NORMAL = 1,
  TASK_PRODUCE_UNBOUND,
  TASK_REDUCE_NORMAL,
  TASK_REDUCE_UNBOUND,
};

struct ProduceNormalTask : public legate::LegateTask<ProduceNormalTask> {
  static const int32_t TASK_ID = TASK_PRODUCE_NORMAL;
  static void cpu_variant(legate::TaskContext& context) {}
};

struct ProduceUnboundTask : public legate::LegateTask<ProduceUnboundTask> {
  static const int32_t TASK_ID = TASK_PRODUCE_UNBOUND;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& output = context.outputs().at(0);
    auto size    = context.get_task_index()[0] + 1;
    auto buffer  = output.create_output_buffer<int64_t, 1>(legate::Point<1>(size), true /*bind*/);
    for (int64_t idx = 0; idx < size; ++idx) buffer[idx] = size;
  }
};

struct ReduceNormalTask : public legate::LegateTask<ReduceNormalTask> {
  static const int32_t TASK_ID = TASK_REDUCE_NORMAL;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs = context.inputs();
    auto& output = context.outputs().at(0);
    for (auto& input : inputs) {
      auto shape = input.shape<1>();
      EXPECT_TRUE(shape.empty() || shape.volume() == TILE_SIZE);
    }
    output.create_output_buffer<int64_t, 1>(legate::Point<1>(0), true);
  }
};

struct ReduceUnboundTask : public legate::LegateTask<ReduceUnboundTask> {
  static const int32_t TASK_ID = TASK_REDUCE_UNBOUND;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs      = context.inputs();
    auto& output      = context.outputs().at(0);
    uint32_t expected = 1;
    for (auto& input : inputs) {
      auto shape = input.shape<1>();
      ASSERT_EQ(shape.volume(), expected);
      ++expected;
    }
    output.create_output_buffer<int64_t, 1>(legate::Point<1>(0), true);
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  ProduceNormalTask::register_variants(context);
  ProduceUnboundTask::register_variants(context);
  ReduceNormalTask::register_variants(context);
  ReduceUnboundTask::register_variants(context);
}

TEST(Integration, TreeReduceNormal)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  size_t num_tasks = 3;
  size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store({num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_NORMAL, {num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_NORMAL, store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST(Integration, TreeReduceNormalTwoSteps)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  size_t num_tasks = 6;
  size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store({num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_NORMAL, {num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_NORMAL, store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST(Integration, TreeReduceUnboud)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  // unbound store
  auto store       = runtime->create_store(legate::int64());
  size_t num_tasks = 4;

  auto task = runtime->create_task(context, TASK_PRODUCE_UNBOUND, {num_tasks});
  task.add_output(store);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_UNBOUND, store, num_tasks);
  EXPECT_FALSE(result.unbound());
}

TEST(Integration, TreeReduceSingleProc)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  // unbound store
  auto store = runtime->create_store(legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_UNBOUND, {1});
  task.add_output(store);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_UNBOUND, store, 4);
  EXPECT_FALSE(result.unbound());
}

}  // namespace tree_reduce
