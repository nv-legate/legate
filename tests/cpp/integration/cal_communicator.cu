/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/comm/coll.h>

#include <gtest/gtest.h>

#include <cal.h>
#include <utilities/utilities.h>

namespace cal_communicator {

namespace {

constexpr std::size_t SIZE = 100;

class CALCommunicatorTester : public legate::LegateTask<CALCommunicatorTester> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{};

  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context)
  {
    if (context.is_single_task()) {
      GTEST_SKIP();
    }

    EXPECT_TRUE(context.communicators().size() == 2);

    auto cal_comm = context.communicators().at(1).get<cal_comm_t>();

    const auto num_tasks = context.get_launch_domain().get_volume();
    int num_ranks;
    const calError_t status = cal_comm_get_size(cal_comm, &num_ranks);
    EXPECT_EQ(status, CAL_OK);
    EXPECT_EQ(num_ranks, num_tasks);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_cal_communicator";
  static void registration_callback(legate::Library library)
  {
    CALCommunicatorTester::register_variants(library);
  }
};

class CALCommunicator : public RegisterOnceFixture<Config> {};

class CalCommunicatorManualTaskTest : public CALCommunicator,
                                      public ::testing::WithParamInterface<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(CALCommunicator,
                         CalCommunicatorManualTaskTest,
                         ::testing::Values(1, 2, 3));

TEST_P(CalCommunicatorManualTaskTest, CAL_communicator)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "No GPUs available. CAL communicator tests require GPUs.";
  }

  legate::Scope scope{machine.only(legate::mapping::TaskTarget::GPU)};

  const auto num_procs = runtime->get_machine().count();
  if (num_procs <= 1) {
    GTEST_SKIP() << "This test requires more than one processor.";
  }

  auto ndim = GetParam();

  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store =
    runtime->create_store(legate::Shape{legate::full<std::uint64_t>(ndim, SIZE)}, legate::int32());
  auto launch_shape = legate::full<std::uint64_t>(ndim, 1);
  auto tile_shape   = legate::full<std::uint64_t>(ndim, 1);
  launch_shape[0]   = num_procs;
  tile_shape[0]     = (SIZE + num_procs - 1) / num_procs;

  auto part = store.partition_by_tiling(tile_shape.data());

  auto task = runtime->create_task(context, CALCommunicatorTester::TASK_ID, launch_shape);
  task.add_output(part);
  task.add_communicator("nccl");
  task.add_communicator("cal");
  runtime->submit(std::move(task));
}

}  // namespace

}  // namespace cal_communicator
