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

namespace broadcast_constraints {

namespace {

static const char* library_name = "test_broadcast_constraints";

constexpr size_t EXT_SMALL = 10;
constexpr size_t EXT_LARGE = 100;

struct TesterTask : public legate::LegateTask<TesterTask> {
  static const int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto shape  = context.outputs().at(0).shape<3>();
    auto extent = context.scalars().at(0).value<uint64_t>();
    auto dims   = context.scalars().at(1).values<int32_t>();

    for (auto dim : dims) {
      EXPECT_EQ(shape.lo[dim], 0);
      EXPECT_EQ(shape.hi[dim], extent - 1);
    }
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  TesterTask::register_variants(context);
}

void test_normal_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto launch_tester = [&](const std::vector<int32_t>& dims) {
    std::vector<size_t> extents(3, EXT_SMALL);
    for (auto dim : dims) extents[dim] = EXT_LARGE;
    auto store = runtime->create_store(extents, legate::int64());
    auto task  = runtime->create_task(context, 0);
    auto part  = task->declare_partition();
    task->add_output(store, part);
    task->add_scalar_arg(legate::Scalar(EXT_LARGE));
    task->add_scalar_arg(legate::Scalar(dims));
    task->add_constraint(legate::broadcast(part, dims));
    runtime->submit(std::move(task));
  };

  launch_tester({0});
  launch_tester({1});
  launch_tester({2});
  launch_tester({0, 1});
  launch_tester({1, 2});
  launch_tester({0, 2});
  launch_tester({0, 1, 2});
}

void test_promoted_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto launch_tester = [&](const int32_t dim) {
    std::vector<size_t> extents(2, EXT_SMALL);
    extents[dim]  = EXT_LARGE;
    auto store    = runtime->create_store(extents, legate::int64());
    auto promoted = store.promote(2, EXT_LARGE);
    auto task     = runtime->create_task(context, 0);
    auto part     = task->declare_partition();
    task->add_output(promoted, part);
    task->add_scalar_arg(legate::Scalar(EXT_LARGE));
    task->add_scalar_arg(legate::Scalar(std::vector<int32_t>{dim}));
    task->add_constraint(legate::broadcast(part, {dim}));
    runtime->submit(std::move(task));
  };

  launch_tester(0);
  launch_tester(1);
}

}  // namespace

TEST(Integration, BroadcastConstraints)
{
  legate::Core::perform_registration<prepare>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  test_normal_store();
  test_promoted_store();
}

}  // namespace broadcast_constraints
