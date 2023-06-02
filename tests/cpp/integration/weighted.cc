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

namespace weighted {

namespace {

const char* library_name = "test_weighted";

enum TaskIDs {
  INIT  = 0,
  CHECK = 3,
};

constexpr uint32_t NUM_TASKS = 4;

struct Initializer : public legate::LegateTask<Initializer> {
  static void cpu_variant(legate::TaskContext& context)
  {
    auto task_idx = context.get_task_index()[0];
    auto& outputs = context.outputs();
    for (uint32_t idx = 0; idx < outputs.size(); ++idx) {
      auto& output = outputs.at(idx);
      output.create_output_buffer<int32_t, 1>(legate::Point<1>(task_idx + 10 * (idx + 1)), true);
    }
  }
};

struct Tester : public legate::LegateTask<Tester> {
  static void cpu_variant(legate::TaskContext& context)
  {
    EXPECT_FALSE(context.is_single_task());
    EXPECT_EQ(context.get_launch_domain().get_volume(), NUM_TASKS);
    auto task_idx = context.get_task_index()[0];
    auto& outputs = context.outputs();
    for (uint32_t idx = 0; idx < outputs.size(); ++idx) {
      auto volume = outputs.at(idx).shape<1>().volume();
      EXPECT_EQ(volume, task_idx + 10 * (idx + 1));
    }
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  Initializer::register_variants(context, INIT);
  Tester::register_variants(context, CHECK);
}

void initialize(legate::Runtime* runtime,
                legate::LibraryContext* context,
                const std::vector<legate::LogicalStore>& outputs)
{
  auto task = runtime->create_task(context, INIT, {NUM_TASKS});

  std::vector<const legate::Variable*> parts;
  for (auto& output : outputs) {
    auto part = task->declare_partition();
    task->add_output(output);
  }

  runtime->submit(std::move(task));
}

void check(legate::Runtime* runtime,
           legate::LibraryContext* context,
           const std::vector<legate::LogicalStore>& inputs)
{
  auto task = runtime->create_task(context, CHECK);

  for (auto& input : inputs) {
    auto part_in  = task->declare_partition();
    auto output   = runtime->create_store(input.extents(), input.type());
    auto part_out = task->declare_partition();
    task->add_input(input, part_in);
    task->add_output(output, part_out);
    task->add_constraint(legate::align(part_in, part_out));
  }

  runtime->submit(std::move(task));
}

void test_weighted(uint32_t num_stores)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  std::vector<legate::LogicalStore> stores;
  for (uint32_t idx = 0; idx < num_stores; ++idx)
    stores.push_back(runtime->create_store(legate::int32()));
  initialize(runtime, context, stores);
  check(runtime, context, stores);
}

}  // namespace

// Test case with single unbound store
TEST(Integration, WeightedSingle)
{
  legate::Core::perform_registration<prepare>();

  test_weighted(1);
}

// Test case with multiple unbound stores
TEST(Integration, WeightedMultiple)
{
  legate::Core::perform_registration<prepare>();

  test_weighted(3);
}

}  // namespace weighted
