/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

constexpr std::uint64_t ALLOCATE_BYTES = 100;
constexpr std::uint64_t OVER_ALIGNMENT = 128;

struct DeallocateTask : public legate::LegateTask<DeallocateTask> {
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext context);
};

struct DoubleDeallocateTask : public legate::LegateTask<DoubleDeallocateTask> {
  static constexpr auto TASK_ID = legate::LocalTaskID{2};

  static void cpu_variant(legate::TaskContext context);
};

struct InvalidAllocateTask : public legate::LegateTask<InvalidAllocateTask> {
  static constexpr auto TASK_ID = legate::LocalTaskID{3};

  static void cpu_variant(legate::TaskContext context);
};

bool check_alignment(const void* buffer, std::size_t alignment)
{
  return alignment == 0 || reinterpret_cast<std::uintptr_t>(buffer) % alignment == 0;
}

/*static*/ void DeallocateTask::cpu_variant(legate::TaskContext context)
{
  auto kind      = context.scalar(0).value<std::uint64_t>();
  auto scoped    = context.scalar(1).value<bool>();
  auto alignment = context.scalar(2).value<std::uint64_t>();
  auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator =
    legate::ScopedAllocator{static_cast<legate::Memory::Kind>(kind), scoped, alignment};
  auto* buffer = allocator.allocate(bytes);

  if (0 == bytes) {
    ASSERT_EQ(buffer, nullptr);
    return;
  }
  ASSERT_TRUE(check_alignment(buffer, alignment));
  ASSERT_NO_THROW(allocator.deallocate(buffer));
}

/*static*/ void DoubleDeallocateTask::cpu_variant(legate::TaskContext context)
{
  auto kind      = context.scalar(0).value<std::uint64_t>();
  auto scoped    = context.scalar(1).value<bool>();
  auto alignment = context.scalar(2).value<std::uint64_t>();
  auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator =
    legate::ScopedAllocator{static_cast<legate::Memory::Kind>(kind), scoped, alignment};
  auto* buffer = allocator.allocate(bytes);

  if (0 == bytes) {
    ASSERT_EQ(buffer, nullptr);
    return;
  }
  ASSERT_TRUE(check_alignment(buffer, alignment));
  ASSERT_NO_THROW(allocator.deallocate(buffer));
  ASSERT_THROW(allocator.deallocate(buffer), std::runtime_error);
}

/*static*/ void InvalidAllocateTask::cpu_variant(legate::TaskContext context)
{
  auto kind      = context.scalar(0).value<std::uint64_t>();
  auto scoped    = context.scalar(1).value<bool>();
  auto alignment = context.scalar(2).value<std::uint64_t>();
  auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator =
    legate::ScopedAllocator{static_cast<legate::Memory::Kind>(kind), scoped, alignment};
  auto* buffer = allocator.allocate(bytes);

  if (0 == bytes) {
    ASSERT_EQ(buffer, nullptr);
    return;
  }
  ASSERT_TRUE(check_alignment(buffer, alignment));
  ASSERT_THROW(allocator.deallocate(static_cast<void*>(static_cast<std::int8_t*>(buffer) + 1)),
               std::runtime_error);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "legate.scopedallocator";
  static void registration_callback(legate::Library library)
  {
    DeallocateTask::register_variants(library);
    DoubleDeallocateTask::register_variants(library);
    InvalidAllocateTask::register_variants(library);
  }
};

class ScopedAllocatorUnit : public RegisterOnceFixture<Config> {};

class ScopedAllocatorTask : public RegisterOnceFixture<Config>,
                            public ::testing::WithParamInterface<
                              std::tuple<std::size_t, std::size_t, legate::Memory::Kind>> {};

// Want to test over-alignment as well.
static_assert(alignof(std::max_align_t) < OVER_ALIGNMENT);

INSTANTIATE_TEST_SUITE_P(
  ScopedAllocatorUnit,
  ScopedAllocatorTask,
  ::testing::Combine(::testing::Values(0, ALLOCATE_BYTES),
                     ::testing::Values(0, 1, alignof(std::max_align_t), OVER_ALIGNMENT),
                     ::testing::Values(legate::Memory::NO_MEMKIND, legate::Memory::SYSTEM_MEM)));
// TODO(joyshennv): issue #1189
//  legate::Memory::GLOBAL_MEM,
//  legate::Memory::REGDMA_MEM,
//  legate::Memory::SOCKET_MEM,
//  legate::Memory::Z_COPY_MEM,
//  legate::Memory::GPU_FB_MEM,
//  legate::Memory::DISK_MEM,
//  legate::Memory::HDF_MEM,
//  legate::Memory::FILE_MEM,
//  legate::Memory::LEVEL3_CACHE,
//  legate::Memory::LEVEL2_CACHE,
//  legate::Memory::LEVEL1_CACHE,
//  legate::Memory::GPU_MANAGED_MEM,
//  legate::Memory::GPU_DYNAMIC_MEM)));

void test_deallocate(legate::LocalTaskID task_id,
                     bool scoped,
                     legate::Memory::Kind kind,
                     std::size_t bytes,
                     std::size_t alignment = alignof(std::max_align_t))
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, task_id);
  auto part    = task.declare_partition();

  static_cast<void>(part);
  task.add_scalar_arg(legate::Scalar{static_cast<std::uint64_t>(kind)});
  task.add_scalar_arg(legate::Scalar{scoped});
  task.add_scalar_arg(legate::Scalar{alignment});
  task.add_scalar_arg(legate::Scalar{bytes});
  runtime->submit(std::move(task));
}

TEST_P(ScopedAllocatorTask, Scoped)
{
  auto& [bytes, alignment, kind] = GetParam();
  test_deallocate(DeallocateTask::TASK_ID, true, kind, bytes, alignment);
}

TEST_P(ScopedAllocatorTask, NotScoped)
{
  auto& [bytes, alignment, kind] = GetParam();
  test_deallocate(DeallocateTask::TASK_ID, false, kind, bytes, alignment);
}

TEST_F(ScopedAllocatorUnit, DoubleDeallocate)
{
  test_deallocate(DoubleDeallocateTask::TASK_ID,
                  true,
                  legate::Memory::SYSTEM_MEM,
                  ALLOCATE_BYTES,
                  alignof(std::max_align_t));
}

TEST_F(ScopedAllocatorUnit, InvalidDeallocateTopLevel)
{
  auto allocator                  = legate::ScopedAllocator{legate::Memory::SYSTEM_MEM, true};
  std::vector<std::uint64_t> data = {1, 2, 3};

  ASSERT_THROW(allocator.deallocate(data.data()), std::runtime_error);
}

TEST_F(ScopedAllocatorUnit, InvalidDeallocate)
{
  test_deallocate(InvalidAllocateTask::TASK_ID,
                  true,
                  legate::Memory::SYSTEM_MEM,
                  ALLOCATE_BYTES,
                  alignof(std::max_align_t));
}

TEST_F(ScopedAllocatorUnit, EmptyAllocate)
{
  auto allocator = legate::ScopedAllocator{legate::Memory::SYSTEM_MEM, true};
  void* ptr      = allocator.allocate(0);

  ASSERT_EQ(ptr, nullptr);
}

TEST_F(ScopedAllocatorUnit, Negative)
{
  // #issue 1170
  // ASSERT_THROW(static_cast<void>(legate::ScopedAllocator(legate::Memory::SYSTEM_MEM, true, -1)),
  // std::invalid_argument);
}

}  // namespace scoped_allocator_test
