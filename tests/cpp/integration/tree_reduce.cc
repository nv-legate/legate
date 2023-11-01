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

#include <gtest/gtest.h>

namespace tree_reduce {

using TreeReduce = DefaultFixture;

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
  static void cpu_variant(legate::TaskContext context) {}
};

struct ProduceUnboundTask : public legate::LegateTask<ProduceUnboundTask> {
  static const int32_t TASK_ID = TASK_PRODUCE_UNBOUND;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto size   = context.get_task_index()[0] + 1;
    auto buffer = output.create_output_buffer<int64_t, 1>(legate::Point<1>(size), true /*bind*/);
    for (int64_t idx = 0; idx < size; ++idx) buffer[idx] = size;
  }
};

struct ReduceNormalTask : public legate::LegateTask<ReduceNormalTask> {
  static const int32_t TASK_ID = TASK_REDUCE_NORMAL;
  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs = context.inputs();
    auto output = context.output(0).data();
    for (auto& input : inputs) {
      auto shape = input.shape<1>();
      EXPECT_TRUE(shape.empty() || shape.volume() == TILE_SIZE);
    }
    output.create_output_buffer<int64_t, 1>(legate::Point<1>(0), true);
  }
};

struct ReduceUnboundTask : public legate::LegateTask<ReduceUnboundTask> {
  static const int32_t TASK_ID = TASK_REDUCE_UNBOUND;
  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs       = context.inputs();
    auto output       = context.output(0).data();
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
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  ProduceNormalTask::register_variants(context);
  ProduceUnboundTask::register_variants(context);
  ReduceNormalTask::register_variants(context);
  ReduceUnboundTask::register_variants(context);
}

TEST_F(TreeReduce, AutoProducer)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  size_t num_tasks = 3;
  size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_NORMAL, legate::Shape{num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_NORMAL, store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducer)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  size_t num_tasks = 3;
  size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_NORMAL, legate::Shape{num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_NORMAL, store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducerMultiLevel)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  size_t num_tasks = 6;
  size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_NORMAL, legate::Shape{num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_NORMAL, store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducerUnbound)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  // unbound store
  auto store       = runtime->create_store(legate::int64());
  size_t num_tasks = 4;

  auto task = runtime->create_task(context, TASK_PRODUCE_UNBOUND, legate::Shape{num_tasks});
  task.add_output(store);
  runtime->submit(std::move(task));

  auto result =
    runtime->tree_reduce(context, TASK_REDUCE_UNBOUND, store, static_cast<std::int64_t>(num_tasks));
  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducerSingle)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  // unbound store
  auto store = runtime->create_store(legate::int64());

  auto task = runtime->create_task(context, TASK_PRODUCE_UNBOUND, legate::Shape{1});
  task.add_output(store);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, TASK_REDUCE_UNBOUND, store, 4);
  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, AutoProducerSingle)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  // unbound store
  auto store = runtime->create_store(legate::int64());

  {
    auto machine = runtime->get_machine();
    legate::MachineTracker tracker(machine.slice(0, 1, legate::mapping::TaskTarget::CPU));
    auto task = runtime->create_task(context, TASK_PRODUCE_UNBOUND);
    task.add_output(store);
    runtime->submit(std::move(task));
  }

  auto result = runtime->tree_reduce(context, TASK_REDUCE_UNBOUND, store, 4);
  EXPECT_FALSE(result.unbound());
}

}  // namespace tree_reduce
