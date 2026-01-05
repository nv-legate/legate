/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/detail/env.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace opaque {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::uint32_t NUM_TASKS = 4;

enum TaskIDs : std::uint8_t {
  INIT  = 0,
  CHECK = 3,
};

struct Initializer : public legate::LegateTask<Initializer> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{INIT}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto task_idx = context.get_task_index()[0];
    auto outputs  = context.outputs();
    for (std::uint32_t idx = 0; idx < outputs.size(); ++idx) {
      auto output = outputs.at(idx).data();
      static_cast<void>(output.create_output_buffer<std::int32_t, 1>(
        legate::Point<1>{task_idx + (10 * (idx + 1))}, /*bind_buffer=*/true));
    }
  }
};

struct Tester : public legate::LegateTask<Tester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CHECK}};

  static void cpu_variant(legate::TaskContext context)
  {
    // Does not work with inline task launches.
    //
    // I think this test is kind of abusing the API. It's trying to make sure that, if you use
    // 4 workers with unbound outputs, the resulting Store will start out partitioned in 4
    // pieces (each piece i containing exactly the elements that worker i produced).
    //
    // Instead of checking that property directly, it is checking that if you launch an
    // AutoTask on that Store it will launch on a 4-worker domain but I can see that not being
    // the case, e.g. if we're running on 1 GPU only, we might want to active fast-path, as if
    // we're running the next task on 1 worker.
    if (legate::detail::experimental::LEGATE_INLINE_TASK_LAUNCH.get(/*default_value=*/false)) {
      return;
    }

    ASSERT_EQ(context.get_launch_domain().get_volume(), NUM_TASKS);

    auto task_idx = context.get_task_index()[0];
    auto outputs  = context.outputs();
    for (std::uint32_t idx = 0; idx < outputs.size(); ++idx) {
      auto volume = outputs.at(idx).shape<1>().volume();
      EXPECT_EQ(volume, task_idx + (10 * (idx + 1)));
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_opaque";

  static void registration_callback(legate::Library library)
  {
    Initializer::register_variants(library);
    Tester::register_variants(library);
  }
};

class Opaque : public RegisterOnceFixture<Config> {};

void initialize(legate::Runtime* runtime,
                legate::Library library,
                const std::vector<legate::LogicalStore>& outputs)
{
  auto task = runtime->create_task(library, Initializer::TASK_CONFIG.task_id(), {NUM_TASKS});

  for (auto& output : outputs) {
    task.add_output(output);
  }

  runtime->submit(std::move(task));
}

void check(legate::Runtime* runtime,
           legate::Library library,
           const std::vector<legate::LogicalStore>& inputs)
{
  auto task = runtime->create_task(library, Tester::TASK_CONFIG.task_id());

  for (auto& input : inputs) {
    auto part_in  = task.add_input(input);
    auto output   = runtime->create_store(input.extents(), input.type());
    auto part_out = task.add_output(output);
    task.add_constraint(legate::align(part_in, part_out));
  }

  runtime->submit(std::move(task));
}

void test_opaque(std::uint32_t num_stores)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  std::vector<legate::LogicalStore> stores;

  stores.reserve(num_stores);
  for (std::uint32_t idx = 0; idx < num_stores; ++idx) {
    stores.push_back(runtime->create_store(legate::int32()));
  }
  initialize(runtime, library, stores);
  check(runtime, library, stores);
}

}  // namespace

// Test case with single unbound store
TEST_F(Opaque, Single) { test_opaque(1); }

// Test case with multiple unbound stores
TEST_F(Opaque, Multiple) { test_opaque(3); }

// NOLINTEND(readability-magic-numbers)

}  // namespace opaque
