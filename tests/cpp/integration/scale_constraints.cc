/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace scale_constraints {

// NOLINTBEGIN(readability-magic-numbers)
namespace {

enum TaskIDs : std::uint8_t {
  SCALE_TESTER,
};

template <std::int32_t DIM>
struct ScaleTester : public legate::LegateTask<ScaleTester<DIM>> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SCALE_TESTER + DIM}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto smaller = context.output(0);
    auto bigger  = context.output(1);

    auto extents = context.scalar(0).values<std::size_t>();
    auto factors = context.scalar(1).values<std::size_t>();

    auto smaller_shape = smaller.shape<DIM>();
    auto bigger_shape  = bigger.shape<DIM>();

    if (bigger_shape.empty()) {
      return;
    }

    for (std::int32_t idx = 0; idx < DIM; ++idx) {
      EXPECT_EQ(smaller_shape.lo[idx] * factors[idx], bigger_shape.lo[idx]);
      auto high =
        context.is_single_task()
          ? extents[idx]
          : std::min<std::int64_t>((smaller_shape.hi[idx] + 1) * factors[idx], extents[idx]);
      EXPECT_EQ(bigger_shape.hi[idx], high - 1);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_scale_constraints";

  static void registration_callback(legate::Library library)
  {
    ScaleTester<1>::register_variants(library);
    ScaleTester<2>::register_variants(library);
    ScaleTester<3>::register_variants(library);
  }
};

class ScaleConstraint : public RegisterOnceFixture<Config> {};

struct ScaleTestSpec {
  legate::tuple<std::uint64_t> factors;
  legate::Shape smaller_extents;
  legate::Shape bigger_extents;
};

void test_scale(const ScaleTestSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto smaller = runtime->create_store(spec.smaller_extents, legate::float16());
  auto bigger  = runtime->create_store(spec.bigger_extents, legate::int64());

  auto task = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SCALE_TESTER) + smaller.dim()});
  auto part_smaller = task.add_output(smaller);
  auto part_bigger  = task.add_output(bigger);
  task.add_constraint(legate::scale(spec.factors, part_smaller, part_bigger));
  task.add_scalar_arg(legate::Scalar{spec.bigger_extents.extents()});
  task.add_scalar_arg(legate::Scalar{spec.factors.data()});

  runtime->submit(std::move(task));
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  {
    auto smaller = runtime->create_store(legate::Shape{1, 2}, legate::float16());
    auto bigger  = runtime->create_store(legate::Shape{2, 3, 4}, legate::int64());

    auto task         = runtime->create_task(context, legate::LocalTaskID{SCALE_TESTER + 2});
    auto part_smaller = task.add_output(smaller);
    auto part_bigger  = task.add_output(bigger);
    task.add_constraint(legate::scale({2, 3}, part_smaller, part_bigger));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto smaller = runtime->create_store(legate::Shape{1, 2, 3}, legate::float16());
    auto bigger  = runtime->create_store(legate::Shape{2, 3, 4}, legate::int64());

    auto task         = runtime->create_task(context, legate::LocalTaskID{SCALE_TESTER + 3});
    auto part_smaller = task.add_output(smaller);
    auto part_bigger  = task.add_output(bigger);
    task.add_constraint(legate::scale({2, 3}, part_smaller, part_bigger));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }
}

}  // namespace

TEST_F(ScaleConstraint, 1D) { test_scale({{3}, {10}, {29}}); }

TEST_F(ScaleConstraint, 2D) { test_scale({{4, 5}, {2, 7}, {10, 30}}); }

TEST_F(ScaleConstraint, 3D) { test_scale({{2, 3, 4}, {5, 5, 5}, {10, 15, 20}}); }

TEST_F(ScaleConstraint, Invalid) { test_invalid(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace scale_constraints
