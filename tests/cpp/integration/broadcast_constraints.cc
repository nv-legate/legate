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

static const char* library_name = "test_broadcast_constraints";

constexpr size_t EXT_SMALL = 10;
constexpr size_t EXT_LARGE = 100;

constexpr int32_t TESTER      = 0;
constexpr int32_t INITIALIZER = 1;

struct TesterTask : public legate::LegateTask<TesterTask> {
  static void cpu_variant(legate::TaskContext& context)
  {
    auto extent  = context.scalars().at(0).value<uint64_t>();
    auto dims    = context.scalars().at(1).values<int32_t>();
    auto is_read = context.scalars().at(2).value<bool>();
    auto shape   = is_read ? context.inputs().at(0).shape<3>() : context.outputs().at(0).shape<3>();

    for (auto dim : dims) {
      EXPECT_EQ(shape.lo[dim], 0);
      EXPECT_EQ(shape.hi[dim], extent - 1);
    }
  }
};

struct Initializer : public legate::LegateTask<Initializer> {
  static void cpu_variant(legate::TaskContext& context) {}
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  TesterTask::register_variants(context, TESTER);
  Initializer::register_variants(context, INITIALIZER);
}

void test_normal_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto launch_tester = [&](const std::vector<int32_t>& dims) {
    std::vector<size_t> extents(3, EXT_SMALL);
    for (auto dim : dims) extents[dim] = EXT_LARGE;
    auto store = runtime->create_store(extents, legate::int64());
    auto task  = runtime->create_task(context, TESTER);
    auto part  = task.declare_partition();
    task.add_output(store, part);
    task.add_scalar_arg(legate::Scalar(EXT_LARGE));
    task.add_scalar_arg(legate::Scalar(dims));
    task.add_scalar_arg(legate::Scalar(false));
    task.add_constraint(legate::broadcast(part, dims));
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

  auto initialize = [&](auto store) {
    auto task = runtime->create_task(context, INITIALIZER);
    auto part = task.declare_partition();
    task.add_output(store, part);
    runtime->submit(std::move(task));
  };

  auto launch_tester = [&](const int32_t dim) {
    std::vector<size_t> extents(2, EXT_SMALL);
    extents[dim] = EXT_LARGE;
    auto store   = runtime->create_store(extents, legate::int64());
    initialize(store);

    auto task = runtime->create_task(context, TESTER);
    auto part = task.declare_partition();
    task.add_input(store.promote(2, EXT_LARGE), part);
    task.add_scalar_arg(legate::Scalar(EXT_LARGE));
    task.add_scalar_arg(legate::Scalar(std::vector<int32_t>{dim}));
    task.add_scalar_arg(legate::Scalar(true));
    task.add_constraint(legate::broadcast(part, {dim}));
    runtime->submit(std::move(task));
  };

  launch_tester(0);
  launch_tester(1);
}

void test_invalid_broadcast()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto task  = runtime->create_task(context, INITIALIZER);
  auto store = runtime->create_store({10}, legate::int64());
  auto part  = task.declare_partition();
  task.add_output(store, part);
  task.add_constraint(legate::broadcast(part, {1}));
  EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
}

TEST(Broadcast, Basic)
{
  legate::Core::perform_registration<prepare>();
  test_normal_store();
}

TEST(Broadcast, WithPromotion)
{
  legate::Core::perform_registration<prepare>();
  test_promoted_store();
}

TEST(Broadcast, Invalid)
{
  legate::Core::perform_registration<prepare>();
  test_invalid_broadcast();
}

}  // namespace broadcast_constraints
