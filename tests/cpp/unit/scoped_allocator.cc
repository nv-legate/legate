/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/type/types.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <utilities/utilities.h>

namespace scoped_allocator_test {

namespace {

constexpr std::uint64_t ALLOCATE_BYTES = 100;
constexpr std::uint64_t OVER_ALIGNMENT = 128;

class DeallocateTask : public legate::LegateTask<DeallocateTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

class DoubleDeallocateTask : public legate::LegateTask<DoubleDeallocateTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

class InvalidAllocateTask : public legate::LegateTask<InvalidAllocateTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

class CopyAllocatorTask : public legate::LegateTask<CopyAllocatorTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{4}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

class AllocateTypeTask : public legate::LegateTask<AllocateTypeTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{5}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

class AllocateAlignedTask : public legate::LegateTask<AllocateAlignedTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{6}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context);
};

bool check_alignment(const void* buffer, std::size_t alignment)
{
  return alignment == 0 || reinterpret_cast<std::uintptr_t>(buffer) % alignment == 0;
}

/*static*/ void DeallocateTask::cpu_variant(legate::TaskContext context)
{
  auto kind      = context.scalar(0).value<legate::Memory::Kind>();
  auto scoped    = context.scalar(1).value<bool>();
  auto alignment = context.scalar(2).value<std::uint64_t>();
  auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator = legate::ScopedAllocator{kind, scoped, alignment};
  auto* buffer   = allocator.allocate(bytes);

  if (0 == bytes) {
    ASSERT_EQ(buffer, nullptr);
    return;
  }
  ASSERT_TRUE(check_alignment(buffer, alignment));
  ASSERT_NO_THROW(allocator.deallocate(buffer));
}

/*static*/ void DoubleDeallocateTask::cpu_variant(legate::TaskContext context)
{
  auto kind      = context.scalar(0).value<legate::Memory::Kind>();
  auto scoped    = context.scalar(1).value<bool>();
  auto alignment = context.scalar(2).value<std::uint64_t>();
  auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator = legate::ScopedAllocator{kind, scoped, alignment};
  auto* buffer   = allocator.allocate(bytes);

  if (0 == bytes) {
    ASSERT_EQ(buffer, nullptr);
    return;
  }
  ASSERT_TRUE(check_alignment(buffer, alignment));
  ASSERT_NO_THROW(allocator.deallocate(buffer));
  ASSERT_THROW(allocator.deallocate(buffer), std::invalid_argument);
}

/*static*/ void InvalidAllocateTask::cpu_variant(legate::TaskContext context)
{
  auto kind      = context.scalar(0).value<legate::Memory::Kind>();
  auto scoped    = context.scalar(1).value<bool>();
  auto alignment = context.scalar(2).value<std::uint64_t>();
  auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator = legate::ScopedAllocator{kind, scoped, alignment};
  auto* buffer   = allocator.allocate(bytes);

  if (0 == bytes) {
    ASSERT_EQ(buffer, nullptr);
    return;
  }
  ASSERT_TRUE(check_alignment(buffer, alignment));
  ASSERT_THROW(allocator.deallocate(static_cast<void*>(static_cast<std::int8_t*>(buffer) + 1)),
               std::invalid_argument);
}

/*static*/ void CopyAllocatorTask::cpu_variant(legate::TaskContext context)
{
  const auto kind      = context.scalar(0).value<legate::Memory::Kind>();
  const auto scoped    = context.scalar(1).value<bool>();
  const auto alignment = context.scalar(2).value<std::uint64_t>();
  const auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator       = legate::ScopedAllocator{kind, scoped, alignment};
  auto allocator_copy  = allocator;
  auto* buffer         = allocator.allocate(bytes);

  ASSERT_NO_THROW(allocator_copy.deallocate(buffer));
}

/*static*/ void AllocateAlignedTask::cpu_variant(legate::TaskContext context)
{
  const auto kind      = context.scalar(0).value<legate::Memory::Kind>();
  const auto scoped    = context.scalar(1).value<bool>();
  const auto alignment = context.scalar(2).value<std::uint64_t>();
  const auto bytes     = context.scalar(3).value<std::uint64_t>();
  auto allocator       = legate::ScopedAllocator{kind, scoped};

  if (alignment > 0 && alignment == (std::uint64_t{1} << static_cast<int>(
                                       std::log2(static_cast<double>(alignment))))) {
    auto* buffer = allocator.allocate_aligned(bytes, alignment);

    ASSERT_TRUE(check_alignment(buffer, alignment));

    if (bytes == 0) {
      ASSERT_TRUE(buffer == nullptr);
    } else {
      ASSERT_TRUE(buffer != nullptr);
    }
    ASSERT_NO_THROW(allocator.deallocate(buffer));
  } else {
    // check bad alignment
    ASSERT_THAT(
      [&] { static_cast<void>(allocator.allocate_aligned(bytes, alignment)); },
      ::testing::ThrowsMessage<std::domain_error>(::testing::HasSubstr("invalid alignment")));
  }
}

namespace {

class AllocateTypeTaskImpl {
 public:
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext context) const
  {
    using T              = legate::type_of_t<CODE>;
    const auto kind      = context.scalar(0).value<legate::Memory::Kind>();
    const auto scoped    = context.scalar(1).value<bool>();
    const auto num_items = context.scalar(2).value<std::uint64_t>();
    auto allocator       = legate::ScopedAllocator{kind, scoped};
    auto* buffer         = allocator.allocate_type<T>(num_items);

    std::fill(buffer, buffer + num_items, T{});
    ASSERT_NO_THROW(allocator.deallocate(buffer));
  }
};

}  // namespace

/*static*/ void AllocateTypeTask::cpu_variant(legate::TaskContext context)
{
  auto code = context.scalar(3).value<legate::Type::Code>();

  legate::type_dispatch(code, AllocateTypeTaskImpl{}, context);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "legate.scopedallocator";

  static void registration_callback(legate::Library library)
  {
    DeallocateTask::register_variants(library);
    DoubleDeallocateTask::register_variants(library);
    InvalidAllocateTask::register_variants(library);
    CopyAllocatorTask::register_variants(library);
    AllocateTypeTask::register_variants(library);
    AllocateAlignedTask::register_variants(library);
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
                     ::testing::Values(1, alignof(std::max_align_t), OVER_ALIGNMENT),
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

  task.add_scalar_arg(legate::Scalar{kind});
  task.add_scalar_arg(legate::Scalar{scoped});
  task.add_scalar_arg(legate::Scalar{alignment});
  task.add_scalar_arg(legate::Scalar{bytes});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(ScopedAllocatorTask, Scoped)
{
  auto& [bytes, alignment, kind] = GetParam();

  test_deallocate(DeallocateTask::TASK_CONFIG.task_id(), /*scoped=*/true, kind, bytes, alignment);
}

TEST_P(ScopedAllocatorTask, NotScoped)
{
  auto& [bytes, alignment, kind] = GetParam();

  test_deallocate(DeallocateTask::TASK_CONFIG.task_id(), /*scoped=*/false, kind, bytes, alignment);
}

TEST_P(ScopedAllocatorTask, CopyScoped)
{
  auto& [bytes, alignment, kind] = GetParam();

  test_deallocate(
    CopyAllocatorTask::TASK_CONFIG.task_id(), /*scoped=*/true, kind, bytes, alignment);
}

TEST_P(ScopedAllocatorTask, CopyNotScoped)
{
  auto& [bytes, alignment, kind] = GetParam();

  test_deallocate(CopyAllocatorTask::TASK_CONFIG.task_id(),
                  /*scoped=*/false,
                  kind,
                  bytes,
                  alignment);
}

TEST_F(ScopedAllocatorUnit, DoubleDeallocate)
{
  test_deallocate(DoubleDeallocateTask::TASK_CONFIG.task_id(),
                  /*scoped=*/true,
                  legate::Memory::SYSTEM_MEM,
                  ALLOCATE_BYTES,
                  alignof(std::max_align_t));
}

TEST_F(ScopedAllocatorUnit, InvalidDeallocateTopLevel)
{
  auto allocator = legate::ScopedAllocator{legate::Memory::SYSTEM_MEM, /*scoped=*/true};
  std::vector<std::uint64_t> data = {1, 2, 3};

  ASSERT_THROW(allocator.deallocate(data.data()), std::invalid_argument);
}

TEST_F(ScopedAllocatorUnit, InvalidDeallocate)
{
  test_deallocate(InvalidAllocateTask::TASK_CONFIG.task_id(),
                  /*scoped=*/true,
                  legate::Memory::SYSTEM_MEM,
                  ALLOCATE_BYTES,
                  alignof(std::max_align_t));
}

TEST_F(ScopedAllocatorUnit, EmptyAllocate)
{
  auto allocator = legate::ScopedAllocator{legate::Memory::SYSTEM_MEM, /*scoped=*/true};
  void* ptr      = allocator.allocate(0);

  ASSERT_EQ(ptr, nullptr);
  ASSERT_NO_THROW(allocator.deallocate(ptr));
}

TEST_F(ScopedAllocatorUnit, BadAlignment)
{
  // -1 is not a power of 2
  ASSERT_THROW(
    static_cast<void>(legate::ScopedAllocator{
      legate::Memory::SYSTEM_MEM, true /* scoped */, static_cast<std::size_t>(-1) /* alignment */}),
    std::domain_error);
  // Not a power of 2
  ASSERT_THROW(static_cast<void>(legate::ScopedAllocator{
                 legate::Memory::SYSTEM_MEM, true /* scoped */, 3 /* alignment */}),
               std::domain_error);
  // Cannot be 0
  ASSERT_THROW(static_cast<void>(legate::ScopedAllocator{
                 legate::Memory::SYSTEM_MEM, true /* scoped */, 0 /* alignment */}),
               std::domain_error);
}

TEST_F(ScopedAllocatorUnit, DeallocateNull)
{
  auto alloc = legate::ScopedAllocator{legate::Memory::SYSTEM_MEM};

  ASSERT_NO_THROW(alloc.deallocate(nullptr));
}

class AllocateType : public RegisterOnceFixture<Config>,
                     public ::testing::WithParamInterface<
                       std::tuple<std::size_t, legate::Type::Code, legate::Memory::Kind>> {};

INSTANTIATE_TEST_SUITE_P(ScopedAllocatorUnit,
                         AllocateType,
                         ::testing::Combine(::testing::Values(0, ALLOCATE_BYTES),
                                            ::testing::Values(legate::Type::Code::BOOL,
                                                              legate::Type::Code::INT8,
                                                              legate::Type::Code::INT16,
                                                              legate::Type::Code::INT32,
                                                              legate::Type::Code::INT64,
                                                              legate::Type::Code::UINT8,
                                                              legate::Type::Code::UINT16,
                                                              legate::Type::Code::UINT32,
                                                              legate::Type::Code::UINT64,
                                                              legate::Type::Code::FLOAT16,
                                                              legate::Type::Code::FLOAT32,
                                                              legate::Type::Code::FLOAT64,
                                                              legate::Type::Code::COMPLEX64,
                                                              legate::Type::Code::COMPLEX128),
                                            ::testing::Values(legate::Memory::NO_MEMKIND,
                                                              legate::Memory::SYSTEM_MEM)));

namespace {

void test_allocate_type(legate::LocalTaskID task_id,
                        bool scoped,
                        legate::Type::Code code,
                        legate::Memory::Kind kind,
                        std::size_t num_items)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, task_id);

  task.add_scalar_arg(legate::Scalar{kind});
  task.add_scalar_arg(legate::Scalar{scoped});
  task.add_scalar_arg(legate::Scalar{num_items});
  task.add_scalar_arg(legate::Scalar{code});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(AllocateType, Scoped)
{
  auto& [num_items, code, kind] = GetParam();

  test_allocate_type(
    AllocateTypeTask::TASK_CONFIG.task_id(), /*scoped=*/true, code, kind, num_items);
}

TEST_P(AllocateType, NotScoped)
{
  auto& [num_items, code, kind] = GetParam();

  test_allocate_type(
    AllocateTypeTask::TASK_CONFIG.task_id(), /*scoped=*/false, code, kind, num_items);
}

class AllocateAligned
  : public RegisterOnceFixture<Config>,
    public ::testing::WithParamInterface<std::tuple<std::size_t, legate::Memory::Kind>> {};

INSTANTIATE_TEST_SUITE_P(
  ScopedAllocatorUnit,
  AllocateAligned,
  ::testing::Combine(::testing::Values(1, 2, 4, alignof(std::max_align_t), OVER_ALIGNMENT),
                     ::testing::Values(legate::Memory::NO_MEMKIND, legate::Memory::SYSTEM_MEM)));

TEST_P(AllocateAligned, CheckAlignment)
{
  auto& [alignment, kind] = GetParam();

  test_deallocate(AllocateAlignedTask::TASK_CONFIG.task_id(),
                  /*scoped=*/true,
                  kind,
                  ALLOCATE_BYTES,
                  alignment);
}

TEST_P(AllocateAligned, ZeroBytesNullptr)
{
  auto& [alignment, kind] = GetParam();

  test_deallocate(AllocateAlignedTask::TASK_CONFIG.task_id(),
                  /*scoped=*/true,
                  kind,
                  /*bytes=*/0,
                  alignment);
}

class AllocateAlignedExceptions
  : public RegisterOnceFixture<Config>,
    public ::testing::WithParamInterface<std::tuple<std::size_t, legate::Memory::Kind>> {};

INSTANTIATE_TEST_SUITE_P(ScopedAllocatorUnit,
                         AllocateAlignedExceptions,
                         ::testing::Combine(::testing::Values(0, 3, 5),
                                            ::testing::Values(legate::Memory::NO_MEMKIND,
                                                              legate::Memory::SYSTEM_MEM)));

TEST_P(AllocateAlignedExceptions, InvalidAlignment)
{
  auto& [alignment, kind] = GetParam();

  test_deallocate(AllocateAlignedTask::TASK_CONFIG.task_id(),
                  /*scoped=*/true,
                  kind,
                  ALLOCATE_BYTES,
                  alignment);
}

}  // namespace scoped_allocator_test
