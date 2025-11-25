/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace tree_reduce {

namespace {

constexpr std::size_t TILE_SIZE = 10;

}  // namespace

enum TaskIDs : std::uint8_t {
  TASK_PRODUCE_NORMAL = 1,
  TASK_PRODUCE_UNBOUND,
  TASK_REDUCE_NORMAL,
  TASK_REDUCE_UNBOUND,
};

struct ProduceNormalTask : public legate::LegateTask<ProduceNormalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_PRODUCE_NORMAL}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

struct ProduceUnboundTask : public legate::LegateTask<ProduceUnboundTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_PRODUCE_UNBOUND}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto size   = context.get_task_index()[0] + 1;
    auto buffer =
      output.create_output_buffer<std::int64_t, 1>(legate::Point<1>{size}, true /*bind*/);
    for (std::int64_t idx = 0; idx < size; ++idx) {
      buffer[idx] = size;
    }
  }
};

struct ReduceNormalTask : public legate::LegateTask<ReduceNormalTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_REDUCE_NORMAL}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs = context.inputs();
    auto output = context.output(0).data();
    for (auto& input : inputs) {
      auto shape = input.shape<1>();
      EXPECT_TRUE(shape.empty() || shape.volume() == TILE_SIZE);
    }
    (void)output.create_output_buffer<std::int64_t, 1>(legate::Point<1>{0}, true);
  }
};

struct ReduceUnboundTask : public legate::LegateTask<ReduceUnboundTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_REDUCE_UNBOUND}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs            = context.inputs();
    auto output            = context.output(0).data();
    std::uint32_t expected = 1;
    for (auto& input : inputs) {
      auto shape = input.shape<1>();
      ASSERT_EQ(shape.volume(), expected);
      ++expected;
    }
    (void)output.create_output_buffer<std::int64_t, 1>(legate::Point<1>{0}, true);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_tree_reduce";

  static void registration_callback(legate::Library library)
  {
    ProduceNormalTask::register_variants(library);
    ProduceUnboundTask::register_variants(library);
    ReduceNormalTask::register_variants(library);
    ReduceUnboundTask::register_variants(library);
  }
};

class TreeReduce : public RegisterOnceFixture<Config> {};

TEST_F(TreeReduce, AutoProducer)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  constexpr std::size_t num_tasks = 3;
  constexpr std::size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, ProduceNormalTask::TASK_CONFIG.task_id(), {num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, ReduceNormalTask::TASK_CONFIG.task_id(), store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducer)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  constexpr std::size_t num_tasks = 3;
  constexpr std::size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, ProduceNormalTask::TASK_CONFIG.task_id(), {num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, ReduceNormalTask::TASK_CONFIG.task_id(), store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducerMultiLevel)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  constexpr std::size_t num_tasks = 6;
  constexpr std::size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task = runtime->create_task(context, ProduceNormalTask::TASK_CONFIG.task_id(), {num_tasks});
  auto part = store.partition_by_tiling({tile_size});
  task.add_output(part);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, ReduceNormalTask::TASK_CONFIG.task_id(), store, 4);

  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducerUnbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  // unbound store
  auto store                      = runtime->create_store(legate::int64());
  constexpr std::size_t num_tasks = 4;

  auto task = runtime->create_task(context, ProduceUnboundTask::TASK_CONFIG.task_id(), {num_tasks});
  task.add_output(store);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(
    context, ReduceUnboundTask::TASK_CONFIG.task_id(), store, static_cast<std::int64_t>(num_tasks));
  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, ManualProducerSingle)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  // unbound store
  auto store = runtime->create_store(legate::int64());

  auto task = runtime->create_task(context, ProduceUnboundTask::TASK_CONFIG.task_id(), {1});
  task.add_output(store);
  runtime->submit(std::move(task));

  auto result = runtime->tree_reduce(context, ReduceUnboundTask::TASK_CONFIG.task_id(), store, 4);
  EXPECT_FALSE(result.unbound());
}

TEST_F(TreeReduce, AutoProducerSingle)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  // unbound store
  auto store = runtime->create_store(legate::int64());

  {
    auto machine = runtime->get_machine();
    const legate::Scope tracker{machine.slice(0, 1, legate::mapping::TaskTarget::CPU)};
    auto task = runtime->create_task(context, ProduceUnboundTask::TASK_CONFIG.task_id());
    task.add_output(store);
    runtime->submit(std::move(task));
  }

  auto result = runtime->tree_reduce(context, ReduceUnboundTask::TASK_CONFIG.task_id(), store, 4);
  EXPECT_FALSE(result.unbound());
}

}  // namespace tree_reduce
