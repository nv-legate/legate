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

namespace buffer_test {

using BufferUnit = DefaultFixture;

static const char* library_name  = "legate.buffer";
constexpr int64_t BUFFER_TASK_ID = 0;

struct BufferParams {
  int32_t dim;
  uint64_t bytes;
  uint64_t kind;
  uint64_t alignment;
};

struct buffer_fn {
  template <int32_t DIM>
  void operator()(uint64_t bytes, uint64_t kind, uint64_t alignment)
  {
    // Todo: add check for memory size after API's ready.
    switch (DIM) {
      case 1: {
        auto buffer = legate::create_buffer<uint64_t>(
          bytes, static_cast<legate::Memory::Kind>(kind), alignment);
        auto memory = buffer.get_instance().get_location();
        EXPECT_TRUE(memory.exists());
        // NO_MEMKIND on a cpu is always mapped to SYSTEM_MEM
        EXPECT_EQ(memory.kind(),
                  kind == legate::Memory::NO_MEMKIND ? legate::Memory::SYSTEM_MEM
                                                     : static_cast<legate::Memory::Kind>(kind));
        break;
      }
      default: {
        auto buffer = legate::create_buffer<uint64_t, DIM>(
          legate::Point<DIM>(bytes), static_cast<legate::Memory::Kind>(kind), alignment);
        auto memory = buffer.get_instance().get_location();
        EXPECT_TRUE(memory.exists());
        // NO_MEMKIND on a cpu is always mapped to SYSTEM_MEM
        EXPECT_EQ(memory.kind(),
                  kind == legate::Memory::NO_MEMKIND ? legate::Memory::SYSTEM_MEM
                                                     : static_cast<legate::Memory::Kind>(kind));
        break;
      }
    }
  }
};

struct BufferTask : public legate::LegateTask<BufferTask> {
  static const int32_t TASK_ID = BUFFER_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void BufferTask::cpu_variant(legate::TaskContext context)
{
  auto buffer_params = context.scalar(0).value<BufferParams>();
  legate::dim_dispatch(buffer_params.dim,
                       buffer_fn{},
                       buffer_params.bytes,
                       buffer_params.kind,
                       buffer_params.alignment);
}

void test_buffer(int32_t dim, uint64_t bytes, legate::Memory::Kind kind, size_t alignment = 16)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto task          = runtime->create_task(context, BUFFER_TASK_ID);
  BufferParams param = {dim, bytes, static_cast<uint64_t>(kind), alignment};
  task.add_scalar_arg(
    legate::Scalar(param,
                   legate::struct_type(
                     true, legate::int32(), legate::uint64(), legate::uint64(), legate::uint64())));
  runtime->submit(std::move(task));
}

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  BufferTask::register_variants(context);
}

TEST_F(BufferUnit, CreateBuffer)
{
  register_tasks();

  // Todo: need to add tests for REGDMA_MEM
  test_buffer(1, 10, legate::Memory::SYSTEM_MEM);
  test_buffer(2, 10, legate::Memory::NO_MEMKIND);
  test_buffer(3, 10, legate::Memory::SYSTEM_MEM);
  test_buffer(4, 10, legate::Memory::SYSTEM_MEM);
}

TEST_F(BufferUnit, NegativeTest)
{
  register_tasks();

  test_buffer(1, 0, legate::Memory::SYSTEM_MEM);
  test_buffer(2, 10, legate::Memory::SYSTEM_MEM, 0);

  // Note: test passes when bytes / alignment set to -1.
  // Todo: Need to add negative test after issue #31 is fixed.
  // test_buffer(3, -1, legate::Memory::NO_MEMKIND);
  // test_buffer(4, 10, legate::Memory::SYSTEM_MEM, -1);
}
}  // namespace buffer_test
