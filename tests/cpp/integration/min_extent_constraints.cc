/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/detail/enumerate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace min_extents_constraints {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::size_t EXT_SMALL = 10;

struct TesterTask : public legate::LegateTask<TesterTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto extents = context.scalar(0).values<std::uint64_t>();
    auto shape1  = context.output(0).shape<3>();
    auto shape2  = context.output(1).shape<3>();

    for (auto&& shape : {shape1, shape2}) {
      for (auto&& [dim, extent] : legate::detail::enumerate(extents)) {
        ASSERT_GE(shape.hi[dim] - shape.lo[dim] + 1, extent);
      }
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_min_extents_constraints";

  static void registration_callback(legate::Library library)
  {
    TesterTask::register_variants(library);
  }
};

class MinExtents : public RegisterOnceFixture<Config> {};

void test_normal_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store1 =
    runtime->create_store(legate::Shape{EXT_SMALL, EXT_SMALL, EXT_SMALL}, legate::int64());
  auto store2 = runtime->create_store(store1.shape(), legate::int64());

  auto launch_tester = [&](const std::vector<std::uint64_t>& min_extents) {
    auto task  = runtime->create_task(context, TesterTask::TASK_CONFIG.task_id());
    auto part1 = task.add_output(store1);
    auto part2 = task.add_output(store2);

    task.add_constraint(legate::min_extents(part1, min_extents));
    task.add_constraint(legate::align(part1, part2));
    task.add_scalar_arg(legate::Scalar{min_extents});
    runtime->submit(std::move(task));
  };

  launch_tester({EXT_SMALL, 0, 0});
  launch_tester({0, EXT_SMALL, 0});
  launch_tester({0, 0, EXT_SMALL});
  launch_tester({EXT_SMALL, EXT_SMALL, 0});
  launch_tester({0, EXT_SMALL, EXT_SMALL});
  launch_tester({EXT_SMALL, 0, EXT_SMALL});
  launch_tester({EXT_SMALL, EXT_SMALL, EXT_SMALL});
}

void test_invalid_min_extents()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto task  = runtime->create_task(context, TesterTask::TASK_CONFIG.task_id());
  auto store = runtime->create_store(legate::Shape{10}, legate::int64());
  auto part  = task.add_output(store);

  task.add_constraint(legate::min_extents(part, {1, 2}));
  EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
}

}  // namespace

TEST_F(MinExtents, Basic) { test_normal_store(); }

TEST_F(MinExtents, Invalid) { test_invalid_min_extents(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace min_extents_constraints
