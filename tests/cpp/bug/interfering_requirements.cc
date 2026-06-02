/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

// This test case exercises a case where an Opaque partition is applied twice to the same index
// space within the same task and the target index space is different from what's recorded in the
// Opaque partition. In this case, the partitioner intersects the target index space with the index
// partition recorded in the Opaque partition. This intersection can return a fresh Legion handle on
// each invocation, so the runtime must deduplicate them to avoid interfering region requirements in
// the task.
namespace interfering_requirements {

namespace {

constexpr std::uint64_t EXT = 42;

class DummyTask : public legate::LegateTask<DummyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0);

    if (output.is_unbound_store()) {
      static_cast<void>(
        output.create_output_buffer(legate::DomainPoint{EXT}, /*bind_buffer=*/true));
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_interfering_requirements";

  static void registration_callback(legate::Library library)
  {
    DummyTask::register_variants(library);
  }
};

}  // namespace

using InterferingRequirements = RegisterOnceFixture<Config>;

TEST_F(InterferingRequirements, OpaqueLeadsToDuplicateLegionPartitions)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // A dummy store to initialization tasks. without it, the tasks aren't parallelized.
  auto dummy = runtime->create_store(legate::Shape{EXT}, legate::int64());
  // Create and initializes an unbound store st1
  auto st1 = runtime->create_store(legate::int64());

  {
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());

    task.add_output(st1);
    task.add_output(dummy);
    runtime->submit(std::move(task));
  }

  // Transfer st1's key partition to another array st2. As a result, st2 gets an Opque partition
  // as its key partition.
  auto st2 = runtime->create_store(legate::Shape{st1.shape()[0]}, legate::int8());
  {
    auto task  = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
    auto part1 = task.add_input(st1);
    auto part2 = task.add_output(st2);

    task.add_output(dummy);
    task.add_constraint(legate::align(part1, part2));
    runtime->submit(std::move(task));
  }

  // Finally, pass st2 twice in conflicting modes to a task. Unless the Opaque partition cached in
  // st2 leads to the same Legion partition, the code will encounter interfering requirements.
  {
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
    task.add_input(st2);
    task.add_output(st2);
    runtime->submit(std::move(task));
  }
}

}  // namespace interfering_requirements
