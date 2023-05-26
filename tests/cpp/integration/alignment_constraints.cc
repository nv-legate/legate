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

namespace alignment_constraints {

namespace {

static const char* library_name = "test_alignment_constraints";

enum TaskIDs {
  INIT                       = 0,
  ALIGNMENT_TESTER           = 0,
  ALIGNMENT_BROADCAST_TESTER = 3,
  TRANSFORMED_TESTER         = 6,
};

// Dummy task to make the runtime think the store is initialized
struct Initializer : public legate::LegateTask<Initializer> {
  static const int32_t TASK_ID = INIT;
  static void cpu_variant(legate::TaskContext& context) {}
};

template <int32_t DIM>
struct AlignmentTester : public legate::LegateTask<AlignmentTester<DIM>> {
  static const int32_t TASK_ID = ALIGNMENT_TESTER + DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& outputs = context.outputs();
    auto shape    = outputs.at(0).shape<DIM>();
    for (auto& output : outputs) EXPECT_EQ(shape, output.shape<DIM>());
  }
};

template <int32_t DIM>
struct AlignmentBroadcastTester : public legate::LegateTask<AlignmentBroadcastTester<DIM>> {
  static const int32_t TASK_ID = ALIGNMENT_BROADCAST_TESTER + DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto shape1 = context.outputs().at(0).shape<DIM>();
    auto shape2 = context.outputs().at(1).shape<DIM>();
    auto extent = context.scalars().at(0).value<size_t>();
    EXPECT_EQ(shape1, shape2);
    EXPECT_EQ(shape1.lo[0], 0);
    EXPECT_EQ(shape1.hi[0] + 1, extent);
    EXPECT_EQ(shape1.hi[0] - shape1.lo[0] + 1, extent);
  }
};

template <int32_t DIM>
struct TransformedTester : public legate::LegateTask<TransformedTester<DIM>> {
  static const int32_t TASK_ID = TRANSFORMED_TESTER + DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto shape1 = context.inputs().at(0).shape<DIM>();
    auto shape2 = context.inputs().at(1).shape<DIM>();
    EXPECT_EQ(shape1, shape2);
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  Initializer::register_variants(context);
  AlignmentTester<1>::register_variants(context);
  AlignmentTester<2>::register_variants(context);
  AlignmentTester<3>::register_variants(context);
  AlignmentBroadcastTester<1>::register_variants(context);
  AlignmentBroadcastTester<2>::register_variants(context);
  AlignmentBroadcastTester<3>::register_variants(context);
  TransformedTester<1>::register_variants(context);
  TransformedTester<2>::register_variants(context);
  TransformedTester<3>::register_variants(context);
}

void test_alignment()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto launch_tester = [&](const std::vector<size_t>& extents) {
    auto store1 = runtime->create_store(extents, legate::int64());
    auto store2 = runtime->create_store(extents, legate::int64());
    auto store3 = runtime->create_store(extents, legate::int64());

    auto task  = runtime->create_task(context, ALIGNMENT_TESTER + extents.size());
    auto part1 = task->declare_partition();
    auto part2 = task->declare_partition();
    auto part3 = task->declare_partition();

    task->add_output(store1, part1);
    task->add_output(store2, part2);
    task->add_output(store3, part3);

    task->add_constraint(legate::align(part1, part2));
    task->add_constraint(legate::align(part2, part3));

    runtime->submit(std::move(task));
  };

  launch_tester({10});
  launch_tester({10, 10});
  launch_tester({10, 10, 10});
}

void test_alignment_and_broadcast()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto launch_tester = [&](const std::vector<size_t>& extents) {
    auto store1 = runtime->create_store(extents, legate::int64());
    auto store2 = runtime->create_store(extents, legate::int64());

    auto task  = runtime->create_task(context, ALIGNMENT_BROADCAST_TESTER + extents.size());
    auto part1 = task->declare_partition();
    auto part2 = task->declare_partition();

    task->add_output(store1, part1);
    task->add_output(store2, part2);

    task->add_scalar_arg(legate::Scalar(extents[0]));

    task->add_constraint(legate::align(part1, part2));
    task->add_constraint(legate::broadcast(part1, {0}));

    runtime->submit(std::move(task));
  };

  launch_tester({100});
  launch_tester({100, 10});
  launch_tester({100, 10, 20});
}

void initialize(legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, INIT);

  auto part = task->declare_partition();
  task->add_output(store, part);

  runtime->submit(std::move(task));
}

void test_alignment_transformed()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto launch_tester = [&](auto store1, auto store2) {
    auto task  = runtime->create_task(context, TRANSFORMED_TESTER + store1.dim());
    auto part1 = task->declare_partition();
    auto part2 = task->declare_partition();

    task->add_input(store1, part1);
    task->add_input(store2, part2);

    task->add_constraint(legate::align(part1, part2));

    runtime->submit(std::move(task));
  };

  // TODO: Extents are chosen such that imaginary dimensions are not partitioned.
  // Such scenarios should be tested once we pass down partition metadata to the tasks
  // and have them compute extents on imaginary dimensions based on that
  auto store1 = runtime->create_store({100}, legate::int64());
  auto store2 = runtime->create_store({100, 10}, legate::int64());
  auto store3 = runtime->create_store({100, 10, 2}, legate::int64());
  auto store4 = runtime->create_store({10, 10}, legate::int64());

  initialize(store1);
  initialize(store2);
  initialize(store3);
  initialize(store4);

  launch_tester(store1.promote(1, 10), store2);
  launch_tester(store1, store2.project(1, 5));
  launch_tester(store1.promote(1, 10), store3.project(2, 0));
  launch_tester(store1.promote(1, 10).promote(2, 2), store3);
  launch_tester(store1.promote(1, 5), store2.slice(1, legate::Slice(3, 8)));
  launch_tester(store1.promote(0, 10).transpose({1, 0}), store2);
  launch_tester(store1.delinearize(0, {10, 10}), store4);
}

}  // namespace

TEST(Integration, AlignmentConstraints)
{
  legate::Core::perform_registration<prepare>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  test_alignment();
  test_alignment_and_broadcast();
  test_alignment_transformed();
}

}  // namespace alignment_constraints
