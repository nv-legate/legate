/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/comm/coll.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace cpu_communicator {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::size_t SIZE = 10;

struct CPUCommunicatorTester : public legate::LegateTask<CPUCommunicatorTester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }
    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    constexpr std::int64_t value = 12345;
    const auto num_tasks         = context.get_launch_domain().get_volume();
    std::vector<std::int64_t> recv_buffer(num_tasks, 0);
    collAllgather(&value, recv_buffer.data(), 1, legate::comm::coll::CollDataType::CollInt64, comm);
    for (auto v : recv_buffer) {
      EXPECT_EQ(v, value);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_cpu_communicator";

  static void registration_callback(legate::Library library)
  {
    CPUCommunicatorTester::register_variants(library);
  }
};

class CPUCommunicator : public RegisterOnceFixture<Config> {};

void test_cpu_communicator_auto(std::int32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());

  auto task = runtime->create_task(context, CPUCommunicatorTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();
  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
}

void test_cpu_communicator_manual(std::int32_t ndim)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto num_procs = runtime->get_machine().count();
  if (num_procs <= 1) {
    return;
  }

  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());
  auto launch_shape = legate::full<std::uint64_t>(ndim, 1);
  auto tile_shape   = legate::full<std::uint64_t>(ndim, 1);
  launch_shape[0]   = num_procs;
  tile_shape[0]     = (SIZE + num_procs - 1) / num_procs;

  auto part = store.partition_by_tiling(tile_shape.data());

  auto task =
    runtime->create_task(context, CPUCommunicatorTester::TASK_CONFIG.task_id(), launch_shape);
  task.add_output(part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
}

}  // namespace

// Test case with single unbound store
// TODO(jfaibussowit)
// Currently causes unexplained hangs in CI. To be fixed by
// https://github.com/nv-legate/legate.internal/pull/700
TEST_F(CPUCommunicator, DISABLED_ALL)
{
  for (auto ndim : {1, 3}) {
    test_cpu_communicator_auto(ndim);
    test_cpu_communicator_manual(ndim);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace cpu_communicator
