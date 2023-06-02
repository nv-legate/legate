/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "legate.h"

#include <nccl.h>

namespace nccl {

namespace {

const char* library_name = "test_nccl";

enum TaskIDs {
  NCCL_TESTER = 0,
};

constexpr size_t SIZE = 10;

struct NCCLTester : public legate::LegateTask<NCCLTester> {
  static void gpu_variant(legate::TaskContext& context)
  {
    EXPECT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) return;
    auto comm = context.communicators().at(0).get<ncclComm_t*>();

    size_t num_tasks = context.get_launch_domain().get_volume();

    auto recv_buffer = legate::create_buffer<uint64_t>(num_tasks, legate::Memory::Kind::Z_COPY_MEM);
    auto send_buffer = legate::create_buffer<uint64_t>(1, legate::Memory::Kind::Z_COPY_MEM);

    auto* p_recv = recv_buffer.ptr(0);
    auto* p_send = send_buffer.ptr(0);

    for (uint32_t idx = 0; idx < num_tasks; ++idx) p_recv[idx] = 0;
    *p_send = 12345;

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto result = ncclAllGather(p_send, p_recv, 1, ncclUint64, *comm, stream);
    EXPECT_EQ(result, ncclSuccess);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (uint32_t idx = 0; idx < num_tasks; ++idx) EXPECT_EQ(p_recv[idx], 12345);
  }
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  NCCLTester::register_variants(context, NCCL_TESTER);
}

void test_nccl_auto(int32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto store   = runtime->create_store(std::vector<size_t>(ndim, SIZE), legate::int32());

  auto task = runtime->create_task(context, NCCL_TESTER);
  auto part = task->declare_partition();
  task->add_output(store, part);
  task->add_communicator("cpu");  // This requested will be ignored
  task->add_communicator("nccl");
  runtime->submit(std::move(task));
}

void test_nccl_manual(int32_t ndim)
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

  auto task = runtime->create_task(context, NCCL_TESTER, std::move(launch_shape));
  task->add_output(part);
  task->add_communicator("cpu");  // This requested will be ignored
  task->add_communicator("nccl");
  runtime->submit(std::move(task));
}

}  // namespace

// Test case with single unbound store
TEST(Integration, NCCL)
{
  legate::Core::perform_registration<prepare>();

  auto runtime  = legate::Runtime::get_runtime();
  auto& machine = runtime->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) return;
  legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::GPU));

  for (int32_t ndim : {1, 3}) {
    test_nccl_auto(ndim);
    test_nccl_manual(ndim);
  }
}

}  // namespace nccl
