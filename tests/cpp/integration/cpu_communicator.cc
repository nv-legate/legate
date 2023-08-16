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

#include "core/comm/coll.h"
#include "legate.h"

namespace cpu_communicator {

const char* library_name = "test_cpu_communicator";

enum TaskIDs {
  CPU_COMM_TESTER = 0,
};

constexpr size_t SIZE = 10;

struct CPUCommunicatorTester : public legate::LegateTask<CPUCommunicatorTester> {
  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) return;
    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    int64_t value    = 12345;
    size_t num_tasks = context.get_launch_domain().get_volume();
    std::vector<int64_t> recv_buffer(num_tasks, 0);
    auto result = collAllgather(
      &value, recv_buffer.data(), 1, legate::comm::coll::CollDataType::CollInt64, comm);
    EXPECT_EQ(result, legate::comm::coll::CollSuccess);
    for (auto v : recv_buffer) EXPECT_EQ(v, 12345);
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  CPUCommunicatorTester::register_variants(context, CPU_COMM_TESTER);
}

void test_cpu_communicator_auto(int32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto store   = runtime->create_store(std::vector<size_t>(ndim, SIZE), legate::int32());

  auto task = runtime->create_task(context, CPU_COMM_TESTER);
  auto part = task.declare_partition();
  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
}

void test_cpu_communicator_manual(int32_t ndim)
{
  auto runtime     = legate::Runtime::get_runtime();
  size_t num_procs = runtime->get_machine().count();
  if (num_procs <= 1) return;

  auto context = runtime->find_library(library_name);
  auto store   = runtime->create_store(std::vector<size_t>(ndim, SIZE), legate::int32());
  std::vector<size_t> launch_shape(ndim, 1);
  std::vector<size_t> tile_shape(ndim, 1);
  launch_shape[0] = num_procs;
  tile_shape[0]   = (SIZE + num_procs - 1) / num_procs;

  auto part = store.partition_by_tiling(std::move(tile_shape));

  auto task = runtime->create_task(context, CPU_COMM_TESTER, std::move(launch_shape));
  task.add_output(part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
}

// Test case with single unbound store
TEST(Integration, CPUCommunicator)
{
  legate::Core::perform_registration<prepare>();

  for (int32_t ndim : {1, 3}) {
    test_cpu_communicator_auto(ndim);
    test_cpu_communicator_manual(ndim);
  }
}

}  // namespace cpu_communicator
