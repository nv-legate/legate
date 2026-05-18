/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace broadcast_constraints {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::size_t EXT_SMALL = 10;
constexpr std::size_t EXT_LARGE = 100;

struct TesterTask : public legate::LegateTask<TesterTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto extent  = context.scalar(0).value<std::uint64_t>();
    auto dims    = context.scalar(1).values<std::uint32_t>();
    auto is_read = context.scalar(2).value<bool>();
    auto shape   = is_read ? context.input(0).shape<3>() : context.output(0).shape<3>();

    for (auto dim : dims) {
      EXPECT_EQ(shape.lo[dim], 0);
      EXPECT_EQ(shape.hi[dim], extent - 1);
    }
  }
};

struct TesterForUnboundStoresTask : public legate::LegateTask<TesterForUnboundStoresTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context)
  {
    // If LEGATE_TEST=1 wasn't set, this test doesn't check what it's designed to do
    if (context.is_single_task()) {
      context.output(0).data().bind_empty_data();
      context.output(1).data().bind_empty_data();
      context.output(2).data().bind_empty_data();
      return;
    }

    const auto num_procs = context.machine().count();
    const auto& domain   = context.get_launch_domain();
    const auto lo        = domain.lo();
    const auto hi        = domain.hi();

    ASSERT_EQ((hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1), num_procs);
    for (std::int32_t dim = 2; dim < domain.dim; ++dim) {
      ASSERT_EQ(lo[dim], 0);
      ASSERT_EQ(hi[dim], 0);
    }

    context.output(0).data().bind_empty_data();
    context.output(1).data().bind_empty_data();
    context.output(2).data().bind_empty_data();
  }
};

struct Initializer : public legate::LegateTask<Initializer> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_broadcast_constraints";

  static void registration_callback(legate::Library library)
  {
    TesterTask::register_variants(library);
    Initializer::register_variants(library);
    TesterForUnboundStoresTask::register_variants(library);
  }
};

class Broadcast : public RegisterOnceFixture<Config> {};

void test_normal_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto launch_tester = [&](const std::vector<std::uint32_t>& dims, bool omit_dims_in_broadcast) {
    std::vector<std::uint64_t> extents(3, EXT_SMALL);

    for (auto dim : dims) {
      extents[dim] = EXT_LARGE;
    }
    auto store = runtime->create_store(legate::Shape{extents}, legate::int64());
    auto task  = runtime->create_task(context, TesterTask::TASK_CONFIG.task_id());
    auto part  = task.add_output(store);
    task.add_scalar_arg(legate::Scalar{EXT_LARGE});
    task.add_scalar_arg(legate::Scalar{dims});
    task.add_scalar_arg(legate::Scalar{false});
    if (omit_dims_in_broadcast) {
      task.add_constraint(legate::broadcast(part));
    } else {
      task.add_constraint(legate::broadcast(part, legate::tuple<std::uint32_t>{dims}));
    }
    runtime->submit(std::move(task));
  };

  launch_tester({0}, false);
  launch_tester({1}, false);
  launch_tester({2}, false);
  launch_tester({0, 1}, false);
  launch_tester({1, 2}, false);
  launch_tester({0, 2}, false);
  launch_tester({0, 1, 2}, false);
  launch_tester({0, 1, 2}, true);
}

void test_promoted_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto initialize = [&](auto&& store) {
    auto task = runtime->create_task(context, Initializer::TASK_CONFIG.task_id());

    task.add_output(store);
    runtime->submit(std::move(task));
  };

  auto launch_tester = [&](const std::uint32_t dim) {
    std::vector<std::uint64_t> extents(2, EXT_SMALL);
    extents[dim] = EXT_LARGE;
    auto store   = runtime->create_store(legate::Shape{extents}, legate::int64());
    initialize(store);

    auto task = runtime->create_task(context, TesterTask::TASK_CONFIG.task_id());
    auto part = task.add_input(store.promote(/*extra_dim=*/2, EXT_LARGE));
    task.add_scalar_arg(legate::Scalar{EXT_LARGE});
    task.add_scalar_arg(legate::Scalar{std::vector<std::uint32_t>{dim}});
    task.add_scalar_arg(legate::Scalar{true});
    task.add_constraint(legate::broadcast(part, legate::tuple<std::uint32_t>{dim}));
    runtime->submit(std::move(task));
  };

  launch_tester(0);
  launch_tester(1);
}

void test_unbound_store()
{
  auto machine = legate::get_machine().only(legate::mapping::TaskTarget::CPU);
  if (machine.count() <= 1) {
    GTEST_SKIP() << "The test doesn't do anything with a single CPU ";
  }

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto unbound1 = runtime->create_store(legate::int64(), /*dim=*/3);
  auto unbound2 = runtime->create_store(legate::int64(), /*dim=*/3);
  auto unbound3 = runtime->create_store(legate::int64(), /*dim=*/4);
  auto extents  = std::vector<std::uint64_t>(4, EXT_SMALL);
  auto bound    = runtime->create_store(legate::Shape{extents}, legate::int64());

  auto task  = runtime->create_task(context, TesterForUnboundStoresTask::TASK_CONFIG.task_id());
  auto part1 = task.add_output(unbound1);
  auto part2 = task.add_output(unbound2);
  auto part3 = task.add_output(unbound3);
  auto part4 = task.add_output(bound);
  task.add_constraint(legate::align(part1, part2));
  task.add_constraint(legate::broadcast(part1, legate::tuple<std::uint32_t>{1}));
  task.add_constraint(legate::broadcast(part3, legate::tuple<std::uint32_t>{0, 2}));
  task.add_constraint(legate::broadcast(part4, legate::tuple<std::uint32_t>{2, 3}));
  runtime->submit(std::move(task));
}

void test_invalid_broadcast()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto task  = runtime->create_task(context, Initializer::TASK_CONFIG.task_id());
  auto store = runtime->create_store(legate::Shape{10}, legate::int64());
  auto part  = task.add_output(store);
  EXPECT_THROW(task.add_constraint(legate::broadcast(part, {})), std::invalid_argument);
  task.add_constraint(legate::broadcast(part, legate::tuple<std::uint32_t>{1}));
  EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
}

}  // namespace

TEST_F(Broadcast, Basic) { test_normal_store(); }

TEST_F(Broadcast, WithPromotion) { test_promoted_store(); }

TEST_F(Broadcast, Unbound) { test_unbound_store(); }

TEST_F(Broadcast, Invalid) { test_invalid_broadcast(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace broadcast_constraints
