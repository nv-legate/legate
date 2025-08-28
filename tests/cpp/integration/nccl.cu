/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/cuda/detail/cuda_driver_api.h>

#include <gtest/gtest.h>

#include <nccl.h>
#include <utilities/utilities.h>

namespace test_nccl {

constexpr std::size_t SIZE = 100;

struct NCCLTester : public legate::LegateTask<NCCLTester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};
  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);

  static void gpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }
    auto comm = context.communicators().at(0).get<ncclComm_t*>();

    std::size_t num_tasks = context.get_launch_domain().get_volume();

    auto recv_buffer =
      legate::create_buffer<std::uint64_t>(num_tasks, legate::Memory::Kind::Z_COPY_MEM);
    auto send_buffer = legate::create_buffer<std::uint64_t>(1, legate::Memory::Kind::Z_COPY_MEM);

    auto* p_recv = recv_buffer.ptr(0);
    auto* p_send = send_buffer.ptr(0);

    for (std::uint32_t idx = 0; idx < num_tasks; ++idx) {
      p_recv[idx] = 0;
    }
    *p_send = 12345;

    auto stream = context.get_task_stream();

    /// [NCCL collective operation]
    // The barrier must happen before the NCCL calls begin
    context.concurrent_task_barrier();
    auto result = ncclAllGather(p_send, p_recv, 1, ncclUint64, *comm, stream);
    EXPECT_EQ(result, ncclSuccess);
    // And insert a barrier after all NCCL calls return, to ensure that all ranks have
    // emitted the NCCL calls
    context.concurrent_task_barrier();
    /// [NCCL collective operation]

    legate::cuda::detail::get_cuda_driver_api()->stream_synchronize(stream);
    for (std::uint32_t idx = 0; idx < num_tasks; ++idx) {
      EXPECT_EQ(p_recv[idx], 12345);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_nccl";

  static void registration_callback(legate::Library library)
  {
    NCCLTester::register_variants(library, NCCLTester::TASK_CONFIG.task_id());
  }
};

class NCCL : public RegisterOnceFixture<Config> {};

void test_nccl_auto(std::int32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(legate::full(ndim, SIZE), legate::int32());

  auto task = runtime->create_task(context, NCCLTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();
  task.add_output(store, part);
  task.add_communicator("nccl");
  runtime->submit(std::move(task));
}

void test_nccl_manual(std::int32_t ndim)
{
  auto runtime          = legate::Runtime::get_runtime();
  std::size_t num_procs = runtime->get_machine().count();
  if (num_procs <= 1) {
    return;
  }

  auto context      = runtime->find_library(Config::LIBRARY_NAME);
  auto store        = runtime->create_store(legate::full(ndim, SIZE), legate::int32());
  auto launch_shape = legate::full<std::uint64_t>(ndim, 1);
  auto tile_shape   = legate::full<std::uint64_t>(ndim, 1);
  launch_shape[0]   = num_procs;
  tile_shape[0]     = (SIZE + num_procs - 1) / num_procs;

  auto part = store.partition_by_tiling(tile_shape.data());

  auto task = runtime->create_task(context, NCCLTester::TASK_CONFIG.task_id(), launch_shape);
  task.add_output(part);
  task.add_communicator("nccl");
  runtime->submit(std::move(task));
}

// Test case with single unbound store
TEST_F(NCCL, Auto)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    return;
  }
  legate::Scope scope{machine.only(legate::mapping::TaskTarget::GPU)};

  for (std::int32_t ndim : {1, 3}) {
    test_nccl_auto(ndim);
  }
}

TEST_F(NCCL, Manual)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    return;
  }
  legate::Scope scope{machine.only(legate::mapping::TaskTarget::GPU)};

  for (std::int32_t ndim : {1, 3}) {
    test_nccl_manual(ndim);
  }
}

}  // namespace test_nccl
