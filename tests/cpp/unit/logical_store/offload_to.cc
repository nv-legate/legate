/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_host_offload {

namespace {

decltype(auto) dbg()
{
  static legate::Logger log{"test_offload"};
  return log.debug();
}

class GPUonlyTask : public legate::LegateTask<GPUonlyTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void gpu_variant(legate::TaskContext)
  {
    // leaving empty for now
    dbg() << "gpu_variant running";
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "TEST_HOST_OFFLOAD_LIB";
  static void registration_callback(legate::Library library)
  {
    GPUonlyTask::register_variants(library);
  }
};

}  // namespace

class OffloadAPI : public RegisterOnceFixture<Config> {};

TEST_F(OffloadAPI, GPUToHostOffload)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto internal_runtime     = legate::detail::Runtime::get_runtime();
  const auto& local_machine = internal_runtime->local_machine();

  if (!local_machine.has_gpus()) {
    GTEST_SKIP() << "No GPUs on the machine to test host offload API";
  }

  const auto num_nodes = internal_runtime->node_count();

  std::size_t total_fbmem = 0;
  for (const auto& [gpu, fbmem] : local_machine.frame_buffers()) {
    total_fbmem += fbmem.capacity();
  }

  dbg() << "TOTAL FBMEM = " << total_fbmem;

  // occupy > 50%  of fbmem so that two tasks can't launch without offloading
  const auto STORE_SIZE = (num_nodes * total_fbmem * 8) / 10;

  auto store1 = runtime->create_store(legate::Shape{STORE_SIZE}, legate::int8());
  auto store2 = runtime->create_store(legate::Shape{STORE_SIZE}, legate::int8());

  dbg() << "Both stores created";

  /// [offload-to-host]

  // This snippet launches two GPU tasks that manipulate two different stores,
  // where each store occupies more than 50% of GPU memory. Runtime can map and
  // schedule both the tasks at the same time. Without offloading the first store,
  // mapping will fail for the second task. Therefore, we insert an `offload_to`
  // call for the first store after submitting the first task and before submitting
  // the second task.
  {
    auto task1 = runtime->create_task(library, GPUonlyTask::TASK_ID);

    task1.add_output(store1);
    runtime->submit(std::move(task1));
  }

  store1.offload_to(legate::mapping::StoreTarget::SYSMEM);

  {
    auto task2 = runtime->create_task(library, GPUonlyTask::TASK_ID);

    task2.add_output(store2);
    runtime->submit(std::move(task2));
  }
  /// [offload-to-host]

  dbg() << "Submitted task-1, store1.offload_to() and task-2";
}

}  // namespace test_host_offload
