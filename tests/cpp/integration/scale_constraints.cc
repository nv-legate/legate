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

#include "core/data/detail/logical_store.h"
#include "legate.h"

namespace scale_constraints {

static const char* library_name = "test_scale_constraints";

static legate::Logger logger(library_name);

enum TaskIDs {
  SCALE_TESTER = 0,
};

template <int32_t DIM>
struct ScaleTester : public legate::LegateTask<ScaleTester<DIM>> {
  static const int32_t TASK_ID = SCALE_TESTER + DIM;
  static void cpu_variant(legate::TaskContext context)
  {
    auto smaller = context.output(0);
    auto bigger  = context.output(1);

    auto extents = context.scalar(0).values<size_t>();
    auto factors = context.scalar(1).values<size_t>();

    auto smaller_shape = smaller.shape<DIM>();
    auto bigger_shape  = bigger.shape<DIM>();

    if (bigger_shape.empty()) return;

    for (int32_t idx = 0; idx < DIM; ++idx) {
      EXPECT_EQ(smaller_shape.lo[idx] * factors[idx], bigger_shape.lo[idx]);
      auto high = context.is_single_task()
                    ? extents[idx]
                    : std::min<int64_t>((smaller_shape.hi[idx] + 1) * factors[idx], extents[idx]);
      EXPECT_EQ(bigger_shape.hi[idx], high - 1);
    }
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  ScaleTester<1>::register_variants(context);
  ScaleTester<2>::register_variants(context);
  ScaleTester<3>::register_variants(context);
}

struct ScaleTestSpec {
  legate::Shape factors;
  std::vector<size_t> smaller_extents;
  std::vector<size_t> bigger_extents;
};

void test_scale(const ScaleTestSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto smaller = runtime->create_store(spec.smaller_extents, legate::float16());
  auto bigger  = runtime->create_store(spec.bigger_extents, legate::int64());

  auto task         = runtime->create_task(context, SCALE_TESTER + smaller.dim());
  auto part_smaller = task.add_output(smaller);
  auto part_bigger  = task.add_output(bigger);
  task.add_constraint(legate::scale(spec.factors, part_smaller, part_bigger));
  task.add_scalar_arg(spec.bigger_extents);
  task.add_scalar_arg(spec.factors.data());

  runtime->submit(std::move(task));
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  {
    auto smaller = runtime->create_store({1, 2}, legate::float16());
    auto bigger  = runtime->create_store({2, 3, 4}, legate::int64());

    auto task         = runtime->create_task(context, SCALE_TESTER + 2);
    auto part_smaller = task.add_output(smaller);
    auto part_bigger  = task.add_output(bigger);
    task.add_constraint(legate::scale({2, 3}, part_smaller, part_bigger));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto smaller = runtime->create_store({1, 2, 3}, legate::float16());
    auto bigger  = runtime->create_store({2, 3, 4}, legate::int64());

    auto task         = runtime->create_task(context, SCALE_TESTER + 3);
    auto part_smaller = task.add_output(smaller);
    auto part_bigger  = task.add_output(bigger);
    task.add_constraint(legate::scale({2, 3}, part_smaller, part_bigger));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }
}

TEST(ScaleConstraint, 1D)
{
  prepare();
  test_scale({{3}, {10}, {29}});
}

TEST(ScaleConstraint, 2D)
{
  prepare();
  test_scale({{4, 5}, {2, 7}, {10, 30}});
}

TEST(ScaleConstraint, 3D)
{
  prepare();
  test_scale({{2, 3, 4}, {5, 5, 5}, {10, 15, 20}});
}

TEST(ScaleConstraint, Invalid)
{
  prepare();
  test_invalid();
}

}  // namespace scale_constraints
