/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace tree_reduce_unique {

namespace {

constexpr std::size_t TILE_SIZE = 10;

}  // namespace

enum TaskIDs : std::uint8_t { TASK_FILL = 1, TASK_UNIQUE, TASK_UNIQUE_REDUCE, TASK_CHECK };

struct FillTask : public legate::LegateTask<FillTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_FILL}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto span   = output.span_write_accessor<std::int64_t, 1>();

    for (legate::coord_t i = 0; i < span.extent(0); ++i) {
      span(i) = i / 2;
    }
  }
};

struct UniqueTask : public legate::LegateTask<UniqueTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_UNIQUE}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto output = context.output(0).data();
    auto span   = input.span_read_accessor<std::int64_t, 1>();
    std::unordered_set<std::int64_t> dedup_set;

    for (legate::coord_t i = 0; i < span.extent(0); ++i) {
      dedup_set.insert(span(i));
    }

    auto result = output.create_output_buffer<std::int64_t, 1>(
      static_cast<legate::coord_t>(dedup_set.size()), true);
    std::int64_t pos = 0;
    for (auto e : dedup_set) {
      result[pos++] = e;
    }
  }
};

struct UniqueReduceTask : public legate::LegateTask<UniqueReduceTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_UNIQUE_REDUCE}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    std::vector<std::pair<legate::AccessorRO<std::int64_t, 1>, legate::Rect<1>>> inputs;
    for (auto& input_arr : context.inputs()) {
      auto shape = input_arr.shape<1>();
      auto acc   = input_arr.data().read_accessor<std::int64_t, 1>(shape);
      inputs.emplace_back(acc, shape);
    }
    std::set<std::int64_t> dedup_set;
    for (auto& pair : inputs) {
      auto& input = pair.first;
      auto& shape = pair.second;
      for (auto idx = shape.lo[0]; idx <= shape.hi[0]; ++idx) {
        dedup_set.insert(input[idx]);
      }
    }

    const std::size_t size = dedup_set.size();
    std::int64_t pos       = 0;
    auto result = output.create_output_buffer<std::int64_t, 1>(legate::Point<1>(size), true);
    for (auto e : dedup_set) {
      result[pos++] = e;
    }
  }
};

struct CheckTask : public legate::LegateTask<CheckTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{TASK_CHECK}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto rect   = input.shape<1>();
    auto volume = rect.volume();
    auto in     = input.read_accessor<std::int64_t, 1>(rect);
    ASSERT_EQ(volume, TILE_SIZE / 2);
    for (std::size_t idx = 0; idx < volume; ++idx) {
      ASSERT_EQ(in[idx], idx);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_tree_reduce_unique";

  static void registration_callback(legate::Library library)
  {
    FillTask::register_variants(library);
    UniqueTask::register_variants(library);
    UniqueReduceTask::register_variants(library);
    CheckTask::register_variants(library);
  }
};

class TreeReduceUnique : public RegisterOnceFixture<Config> {};

TEST_F(TreeReduceUnique, All)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  const std::size_t num_tasks = 6;
  const std::size_t tile_size = TILE_SIZE;

  auto store = runtime->create_store(legate::Shape{num_tasks * tile_size}, legate::int64());

  auto task_fill = runtime->create_task(context, FillTask::TASK_CONFIG.task_id(), {num_tasks});
  auto part      = store.partition_by_tiling({tile_size});
  task_fill.add_output(part);
  runtime->submit(std::move(task_fill));

  auto task_unique = runtime->create_task(context, UniqueTask::TASK_CONFIG.task_id(), {num_tasks});
  auto intermediate_result = runtime->create_store(legate::int64(), 1);
  task_unique.add_input(part);
  task_unique.add_output(intermediate_result);
  runtime->submit(std::move(task_unique));

  auto result =
    runtime->tree_reduce(context, UniqueReduceTask::TASK_CONFIG.task_id(), intermediate_result, 4);

  EXPECT_FALSE(result.unbound());

  auto task_check = runtime->create_task(context, CheckTask::TASK_CONFIG.task_id(), {1});
  task_check.add_input(result);
  runtime->submit(std::move(task_check));
}

}  // namespace tree_reduce_unique
