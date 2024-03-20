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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>
#if LegateDefined(LEGATE_USE_CUDA)
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"

#include <cuda_runtime.h>
#endif

namespace replicated_write_test {

// NOLINTBEGIN(readability-magic-numbers)

using ReplicatedWrite = DefaultFixture;

namespace {

constexpr const char library_name[] = "test_replicated_write";

}  // namespace

struct WriterTask : public legate::LegateTask<WriterTask> {
  static constexpr std::int32_t TASK_ID = 0;

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto& output : outputs) {
      auto shape = output.shape<2>();
      auto acc   = output.data().write_accessor<int64_t, 2>();
      for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
        acc[*it] = 42;
      }
    }
  }
#if LegateDefined(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto& output : outputs) {
      auto shape  = output.shape<2>();
      auto acc    = output.data().write_accessor<int64_t, 2>();
      auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
      auto vals   = std::vector<std::int64_t>(shape.volume(), 42);
      CHECK_CUDA(cudaMemcpyAsync(acc.ptr(shape),
                                 vals.data(),
                                 sizeof(int64_t) * shape.volume(),
                                 cudaMemcpyHostToDevice,
                                 stream));
    }
  }
#endif
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  WriterTask::register_variants(library);
}

void validate_output(const legate::LogicalStore& store)
{
  auto p_store = store.get_physical_store();
  auto shape   = p_store.shape<2>();
  auto acc     = p_store.read_accessor<int64_t, 2>();
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    EXPECT_EQ(acc[*it], 42);
  }
}

void test_auto_task(legate::Library library,
                    const legate::tuple<std::uint64_t>& extents,
                    std::uint32_t num_out_stores)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto in_store = runtime->create_store({8, 8}, legate::int64());

  runtime->issue_fill(in_store, legate::Scalar{int64_t{1}});

  auto task = runtime->create_task(library, WriterTask::TASK_ID);

  task.add_input(in_store);

  std::vector<legate::LogicalStore> out_stores;
  for (std::uint32_t idx = 0; idx < num_out_stores; ++idx) {
    auto& out_store = out_stores.emplace_back(
      runtime->create_store(extents, legate::int64(), true /*optimize_scalar*/));
    auto part = task.add_output(out_store);
    task.add_constraint(legate::broadcast(part));
  }
  runtime->submit(std::move(task));

  for (auto&& out_store : out_stores) {
    validate_output(out_store);
  }
}

void test_manual_task(legate::Library library,
                      const legate::tuple<std::uint64_t>& extents,
                      std::uint32_t num_out_stores)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task =
    runtime->create_task(library, WriterTask::TASK_ID, legate::tuple<std::uint64_t>{3, 3});

  std::vector<legate::LogicalStore> out_stores;
  for (std::uint32_t idx = 0; idx < num_out_stores; ++idx) {
    auto& out_store = out_stores.emplace_back(
      runtime->create_store(extents, legate::int64(), true /*optimize_scalar*/));
    task.add_output(out_store);
  }
  runtime->submit(std::move(task));

  for (auto&& out_store : out_stores) {
    validate_output(out_store);
  }
}

TEST_F(ReplicatedWrite, AutoNonScalar)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);
  test_auto_task(library, {2, 2}, 1);
  test_auto_task(library, {3, 3}, 2);
}

TEST_F(ReplicatedWrite, ManualNonScalar)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);
  test_manual_task(library, {2, 2}, 1);
  test_manual_task(library, {3, 3}, 2);
}

TEST_F(ReplicatedWrite, AutoScalar)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);
  test_auto_task(library, {1, 1}, 1);
  test_auto_task(library, {1, 1}, 2);
}

TEST_F(ReplicatedWrite, ManualScalar)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);
  test_manual_task(library, {1, 1}, 1);
  test_manual_task(library, {1, 1}, 2);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace replicated_write_test
