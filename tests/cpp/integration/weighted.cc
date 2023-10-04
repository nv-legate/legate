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

namespace weighted {

const char* library_name = "test_weighted";

enum TaskIDs {
  INIT  = 0,
  CHECK = 3,
};

constexpr uint32_t NUM_TASKS = 4;

struct Initializer : public legate::LegateTask<Initializer> {
  static void cpu_variant(legate::TaskContext context)
  {
    auto task_idx = context.get_task_index()[0];
    auto outputs  = context.outputs();
    for (uint32_t idx = 0; idx < outputs.size(); ++idx) {
      auto output = outputs.at(idx).data();
      output.create_output_buffer<int32_t, 1>(legate::Point<1>(task_idx + 10 * (idx + 1)), true);
    }
  }
};

struct Tester : public legate::LegateTask<Tester> {
  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_FALSE(context.is_single_task());
    EXPECT_EQ(context.get_launch_domain().get_volume(), NUM_TASKS);
    auto task_idx = context.get_task_index()[0];
    auto outputs  = context.outputs();
    for (uint32_t idx = 0; idx < outputs.size(); ++idx) {
      auto volume = outputs.at(idx).shape<1>().volume();
      EXPECT_EQ(volume, task_idx + 10 * (idx + 1));
    }
  }
};

void prepare()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Initializer::register_variants(library, INIT);
  Tester::register_variants(library, CHECK);
}

void initialize(legate::Runtime* runtime,
                legate::Library library,
                const std::vector<legate::LogicalStore>& outputs)
{
  auto task = runtime->create_task(library, INIT, {NUM_TASKS});

  std::vector<const legate::Variable*> parts;
  for (auto& output : outputs) task.add_output(output);

  runtime->submit(std::move(task));
}

void check(legate::Runtime* runtime,
           legate::Library library,
           const std::vector<legate::LogicalStore>& inputs)
{
  auto task = runtime->create_task(library, CHECK);

  for (auto& input : inputs) {
    auto part_in  = task.add_input(input);
    auto output   = runtime->create_store(input.extents(), input.type());
    auto part_out = task.add_output(output);
    task.add_constraint(legate::align(part_in, part_out));
  }

  runtime->submit(std::move(task));
}

void test_weighted(uint32_t num_stores)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  std::vector<legate::LogicalStore> stores;
  for (uint32_t idx = 0; idx < num_stores; ++idx)
    stores.push_back(runtime->create_store(legate::int32()));
  initialize(runtime, library, stores);
  check(runtime, library, stores);
}

// Test case with single unbound store
TEST(Integration, WeightedSingle)
{
  prepare();

  test_weighted(1);
}

// Test case with multiple unbound stores
TEST(Integration, WeightedMultiple)
{
  prepare();

  test_weighted(3);
}

}  // namespace weighted
