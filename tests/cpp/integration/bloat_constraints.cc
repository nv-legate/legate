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

namespace bloat_constraints {

static const char* library_name = "test_bloat_constraints";

static legate::Logger logger(library_name);

enum TaskIDs {
  BLOAT_TESTER = 0,
};

template <int32_t DIM>
struct BloatTester : public legate::LegateTask<BloatTester<DIM>> {
  static const int32_t TASK_ID = BLOAT_TESTER + DIM;
  static void cpu_variant(legate::TaskContext context)
  {
    auto source  = context.input(0);
    auto bloated = context.input(1);

    auto extents      = context.scalar(0).values<size_t>();
    auto low_offsets  = context.scalar(1).values<size_t>();
    auto high_offsets = context.scalar(2).values<size_t>();

    auto source_shape  = source.shape<DIM>();
    auto bloated_shape = bloated.shape<DIM>();

    if (source_shape.empty()) return;

    for (int32_t idx = 0; idx < DIM; ++idx) {
      auto low =
        std::max<int64_t>(0, source_shape.lo[idx] - static_cast<int64_t>(low_offsets[idx]));
      auto high = std::min<int64_t>(static_cast<int64_t>(extents[idx] - 1),
                                    source_shape.hi[idx] + static_cast<int64_t>(high_offsets[idx]));
      EXPECT_EQ(low, bloated_shape.lo[idx]);
      EXPECT_EQ(high, bloated_shape.hi[idx]);
    }
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  BloatTester<1>::register_variants(context);
  BloatTester<2>::register_variants(context);
  BloatTester<3>::register_variants(context);
}

struct BloatTestSpec {
  legate::Shape extents;
  legate::Shape low_offsets;
  legate::Shape high_offsets;
};

void test_bloat(const BloatTestSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto source  = runtime->create_store(spec.extents, legate::int64());
  auto bloated = runtime->create_store(spec.extents, legate::int64());
  runtime->issue_fill(source, legate::Scalar(int64_t{0}));
  runtime->issue_fill(bloated, legate::Scalar(int64_t{0}));

  auto task         = runtime->create_task(context, BLOAT_TESTER + source.dim());
  auto part_source  = task.add_input(source);
  auto part_bloated = task.add_input(bloated);
  task.add_constraint(
    legate::bloat(part_source, part_bloated, spec.low_offsets, spec.high_offsets));
  task.add_scalar_arg(legate::Scalar(spec.extents.data()));
  task.add_scalar_arg(legate::Scalar(spec.low_offsets.data()));
  task.add_scalar_arg(legate::Scalar(spec.high_offsets.data()));

  runtime->submit(std::move(task));
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  {
    auto source  = runtime->create_store({1, 2}, legate::float16());
    auto bloated = runtime->create_store({2, 3, 4}, legate::int64());

    auto task         = runtime->create_task(context, BLOAT_TESTER + 2);
    auto part_source  = task.add_output(source);
    auto part_bloated = task.add_output(bloated);
    task.add_constraint(legate::bloat(part_source, part_bloated, {2, 3}, {4, 5}));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto source  = runtime->create_store({1, 2}, legate::float16());
    auto bloated = runtime->create_store({2, 3}, legate::int64());

    auto task         = runtime->create_task(context, BLOAT_TESTER + 2);
    auto part_source  = task.add_output(source);
    auto part_bloated = task.add_output(bloated);
    task.add_constraint(legate::bloat(part_source, part_bloated, {2, 3, 3}, {4, 5}));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto source  = runtime->create_store({1, 2}, legate::float16());
    auto bloated = runtime->create_store({2, 3}, legate::int64());

    auto task         = runtime->create_task(context, BLOAT_TESTER + 2);
    auto part_source  = task.add_output(source);
    auto part_bloated = task.add_output(bloated);
    task.add_constraint(legate::bloat(part_source, part_bloated, {2, 3}, {4, 5, 3}));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }
}

TEST(BloatConstraint, 1D)
{
  legate::Core::perform_registration<prepare>();
  test_bloat({{10}, {2}, {4}});
}

TEST(BloatConstraint, 2D)
{
  legate::Core::perform_registration<prepare>();
  test_bloat({{9, 9}, {2, 3}, {3, 4}});
}

TEST(BloatConstraint, 3D)
{
  legate::Core::perform_registration<prepare>();
  test_bloat({{10, 10, 10}, {2, 3, 4}, {4, 3, 2}});
}

TEST(BloatConstraint, Invalid)
{
  legate::Core::perform_registration<prepare>();
  test_invalid();
}

}  // namespace bloat_constraints
