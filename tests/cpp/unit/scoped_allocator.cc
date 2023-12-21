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

namespace scoped_allocator_test {

using ScopedAllocatorUnit = DefaultFixture;

static const char* library_name     = "legate.scopedallocator";
constexpr int64_t ALLOCATOR_TASK_ID = 1;

struct AllocatorParams {
  uint32_t op_code;
  uint64_t kind;
  bool scoped;
  uint64_t alignment;
  uint64_t bytes;
};

enum class BufferOpCode : uint32_t {
  DEALLOCATE         = 0,
  DOUBLE_DEALLOCATE  = 1,
  INVALID_DEALLOCATE = 2,
};

struct ScopedAllocatorTask : public legate::LegateTask<ScopedAllocatorTask> {
  static const int32_t TASK_ID = ALLOCATOR_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void ScopedAllocatorTask::cpu_variant(legate::TaskContext context)
{
  auto allocator_params = context.scalar(0).value<AllocatorParams>();
  auto allocator = legate::ScopedAllocator(static_cast<legate::Memory::Kind>(allocator_params.kind),
                                           allocator_params.scoped,
                                           allocator_params.alignment);
  auto buffer    = allocator.allocate(allocator_params.bytes);
  if (0 == allocator_params.bytes) {
    EXPECT_EQ(buffer, nullptr);
  } else {
    EXPECT_NE(buffer, nullptr);
  }

  BufferOpCode code = static_cast<BufferOpCode>(allocator_params.op_code);
  switch (code) {
    case BufferOpCode::DEALLOCATE: {
      if (0 == allocator_params.bytes) {
        EXPECT_THROW(allocator.deallocate(buffer), std::runtime_error);
      } else {
        allocator.deallocate(buffer);
      }
      break;
    }
    case BufferOpCode::DOUBLE_DEALLOCATE: {
      allocator.deallocate(buffer);
      EXPECT_THROW(allocator.deallocate(buffer), std::runtime_error);
      break;
    }
    case BufferOpCode::INVALID_DEALLOCATE: {
      EXPECT_THROW(allocator.deallocate(static_cast<void*>(static_cast<int8_t*>(buffer) + 1)),
                   std::runtime_error);
      break;
    }
  }
}

void test_allocator(
  BufferOpCode op_code, legate::Memory::Kind kind, bool scoped, size_t bytes, size_t alignment = 16)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, ALLOCATOR_TASK_ID);
  auto part    = task.declare_partition();
  static_cast<void>(part);
  AllocatorParams struct_data = {
    static_cast<uint32_t>(op_code), static_cast<uint64_t>(kind), scoped, alignment, bytes};
  task.add_scalar_arg(legate::Scalar(struct_data,
                                     legate::struct_type(true,
                                                         legate::uint32(),
                                                         legate::uint64(),
                                                         legate::bool_(),
                                                         legate::uint64(),
                                                         legate::uint64())));
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
  ScopedAllocatorTask::register_variants(context);
}

TEST_F(ScopedAllocatorUnit, EmptyAllocate)
{
  auto allocator = legate::ScopedAllocator(legate::Memory::SYSTEM_MEM, true);
  void* ptr      = allocator.allocate(0);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(ScopedAllocatorUnit, Allocate)
{
  register_tasks();
  test_allocator(BufferOpCode::DEALLOCATE, legate::Memory::SYSTEM_MEM, true, 1000, 16);
  test_allocator(BufferOpCode::DEALLOCATE, legate::Memory::NO_MEMKIND, true, 1000, 16);
  test_allocator(BufferOpCode::DEALLOCATE, legate::Memory::SYSTEM_MEM, false, 0, 16);
  test_allocator(BufferOpCode::DEALLOCATE, legate::Memory::SYSTEM_MEM, false, 1000, 0);

  // Note: test passes when bytes / alignment set to -1.
  // Todo: Need to add negative test after issue #31 is fixed.
  // test_allocator(BufferOpCode::DEALLOCATE, legate::Memory::SYSTEM_MEM, false, -1, 16);
  // test_allocator(BufferOpCode::DEALLOCATE, legate::Memory::SYSTEM_MEM, false, 1000, -1);
}

TEST_F(ScopedAllocatorUnit, DoubleDeallocate)
{
  register_tasks();
  test_allocator(BufferOpCode::DOUBLE_DEALLOCATE, legate::Memory::SYSTEM_MEM, true, 1000);
}

TEST_F(ScopedAllocatorUnit, InvalidDeallocate)
{
  auto allocator             = legate::ScopedAllocator(legate::Memory::SYSTEM_MEM, true);
  std::vector<uint64_t> data = {1, 2, 3};
  EXPECT_THROW(allocator.deallocate(data.data()), std::runtime_error);

  // invalid deallocate in task launch
  register_tasks();
  test_allocator(BufferOpCode::INVALID_DEALLOCATE, legate::Memory::SYSTEM_MEM, true, 1000);
}
}  // namespace scoped_allocator_test
