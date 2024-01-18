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

namespace tree_reduce_unique {

using TreeReduce = DefaultFixture;

static const char* library_name = "test_tree_reduce_unique";

static const size_t TILE_SIZE = 10;

enum TaskIDs { TASK_FILL = 1, TASK_UNIQUE, TASK_UNIQUE_REDUCE, TASK_CHECK };

struct FillTask : public legate::LegateTask<FillTask> {
  static const int32_t TASK_ID = TASK_FILL;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto rect   = output.shape<1>();
    auto volume = rect.volume();
    auto out    = output.write_accessor<int64_t, 1>(rect);
    for (size_t idx = 0; idx < volume; ++idx) {
      out[idx] = idx / 2;
    }
  }
};

struct UniqueTask : public legate::LegateTask<UniqueTask> {
  static const int32_t TASK_ID = TASK_UNIQUE;
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto output = context.output(0).data();
    auto rect   = input.shape<1>();
    auto volume = rect.volume();
    auto in     = input.read_accessor<int64_t, 1>(rect);
    std::set<int64_t> dedup_set;
    for (size_t idx = 0; idx < volume; ++idx) {
      dedup_set.insert(in[idx]);
    }

    auto result = output.create_output_buffer<int64_t, 1>(dedup_set.size(), true);
    size_t pos  = 0;
    for (auto e : dedup_set) {
      result[pos++] = e;
    }
  }
};

struct UniqueReduceTask : public legate::LegateTask<UniqueReduceTask> {
  static const int32_t TASK_ID = TASK_UNIQUE_REDUCE;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    std::vector<std::pair<legate::AccessorRO<int64_t, 1>, legate::Rect<1>>> inputs;
    for (auto& input_arr : context.inputs()) {
      auto shape = input_arr.shape<1>();
      auto acc   = input_arr.data().read_accessor<int64_t, 1>(shape);
      inputs.push_back(std::make_pair(acc, shape));
    }
    std::set<int64_t> dedup_set;
    for (auto& pair : inputs) {
      auto& input = pair.first;
      auto& shape = pair.second;
      for (auto idx = shape.lo[0]; idx <= shape.hi[0]; ++idx) {
        dedup_set.insert(input[idx]);
      }
    }

    size_t size = dedup_set.size();
    size_t pos  = 0;
    auto result = output.create_output_buffer<int64_t, 1>(legate::Point<1>(size), true);
    for (auto e : dedup_set) {
      result[pos++] = e;
    }
  }
};

struct CheckTask : public legate::LegateTask<CheckTask> {
  static const int32_t TASK_ID = TASK_CHECK;
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto rect   = input.shape<1>();
    auto volume = rect.volume();
    auto in     = input.read_accessor<int64_t, 1>(rect);
    ASSERT_EQ(volume, TILE_SIZE / 2);
    for (size_t idx = 0; idx < volume; ++idx) {
      ASSERT_EQ(in[idx], idx);
    }
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  FillTask::register_variants(context);
  UniqueTask::register_variants(context);
  UniqueReduceTask::register_variants(context);
  CheckTask::register_variants(context);
}

TEST_F(TreeReduce, Unique)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  size_t num_tasks = 6;
  size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task_fill = runtime->create_task(context, TASK_FILL, {num_tasks});
  auto part      = store.partition_by_tiling({tile_size});
  task_fill.add_output(part);
  runtime->submit(std::move(task_fill));

  auto task_unique         = runtime->create_task(context, TASK_UNIQUE, {num_tasks});
  auto intermediate_result = runtime->create_store(legate::int64(), 1);
  task_unique.add_input(part);
  task_unique.add_output(intermediate_result);
  runtime->submit(std::move(task_unique));

  auto result = runtime->tree_reduce(context, TASK_UNIQUE_REDUCE, intermediate_result, 4);

  EXPECT_FALSE(result.unbound());

  auto task_check = runtime->create_task(context, TASK_CHECK, {1});
  task_check.add_input(result);
  runtime->submit(std::move(task_check));
}

}  // namespace tree_reduce_unique
