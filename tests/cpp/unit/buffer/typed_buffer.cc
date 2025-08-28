/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace buffer_test {

namespace {

constexpr std::uint64_t ALLOCATE_BYTES = 10;
constexpr std::uint64_t OVER_ALIGNMENT = 128;

class BufferFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(std::uint64_t bytes, legate::Memory::Kind kind, std::uint64_t alignment) const
  {
    using T = legate::type_of_t<CODE>;

    switch (DIM) {
      case 1: {
        auto buffer = legate::create_buffer<T>(bytes, kind, alignment);
        auto memory = buffer.get_instance().get_location();
        EXPECT_TRUE(memory.exists());
        // NO_MEMKIND on a cpu is always mapped to SYSTEM_MEM
        EXPECT_EQ(memory.kind(),
                  kind == legate::Memory::NO_MEMKIND ? legate::Memory::SYSTEM_MEM
                                                     : static_cast<legate::Memory::Kind>(kind));
        break;
      }
      default: {
        auto buffer = legate::create_buffer<T, DIM>(legate::Point<DIM>{bytes}, kind, alignment);
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

class BufferTask : public legate::LegateTask<BufferTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void BufferTask::cpu_variant(legate::TaskContext context)
{
  auto dim       = context.scalar(0).value<std::int32_t>();
  auto bytes     = context.scalar(1).value<std::uint64_t>();
  auto kind      = context.scalar(2).value<legate::Memory::Kind>();
  auto alignment = context.scalar(3).value<std::size_t>();
  auto code      = context.scalar(4).value<legate::Type::Code>();

  legate::double_dispatch(dim, code, BufferFn{}, bytes, kind, alignment);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "legate.buffer";

  static void registration_callback(legate::Library library)
  {
    BufferTask::register_variants(library);
  }
};

class BufferUnit : public RegisterOnceFixture<Config> {};

class BufferTaskTest
  : public RegisterOnceFixture<Config>,
    public ::testing::WithParamInterface<
      std::tuple<int, std::size_t, legate::Memory::Kind, std::size_t, legate::Type::Code>> {};

INSTANTIATE_TEST_SUITE_P(
  BufferUnit,
  BufferTaskTest,
  ::testing::Values(
    std::make_tuple(1, 0, legate::Memory::NO_MEMKIND, 1, legate::Type::Code::BOOL),
    std::make_tuple(2,
                    ALLOCATE_BYTES,
                    legate::Memory::NO_MEMKIND,
                    alignof(std::max_align_t),
                    legate::Type::Code::INT8),
    std::make_tuple(3,
                    ALLOCATE_BYTES,
                    legate::Memory::NO_MEMKIND,
                    alignof(std::max_align_t),
                    legate::Type::Code::INT16),
    std::make_tuple(LEGATE_MAX_DIM, 0, legate::Memory::NO_MEMKIND, 1, legate::Type::Code::INT32),
    std::make_tuple(1, 0, legate::Memory::NO_MEMKIND, 1, legate::Type::Code::INT64),
    std::make_tuple(2,
                    ALLOCATE_BYTES,
                    legate::Memory::NO_MEMKIND,
                    alignof(std::max_align_t),
                    legate::Type::Code::UINT8),
    std::make_tuple(3,
                    ALLOCATE_BYTES,
                    legate::Memory::NO_MEMKIND,
                    alignof(std::max_align_t),
                    legate::Type::Code::UINT16),
    std::make_tuple(LEGATE_MAX_DIM, 0, legate::Memory::SYSTEM_MEM, 1, legate::Type::Code::UINT32),
    std::make_tuple(1, 0, legate::Memory::SYSTEM_MEM, 1, legate::Type::Code::UINT64),
    std::make_tuple(2,
                    ALLOCATE_BYTES,
                    legate::Memory::SYSTEM_MEM,
                    alignof(std::max_align_t),
                    legate::Type::Code::FLOAT16),
    std::make_tuple(3,
                    ALLOCATE_BYTES,
                    legate::Memory::SYSTEM_MEM,
                    alignof(std::max_align_t),
                    legate::Type::Code::FLOAT32),
    std::make_tuple(LEGATE_MAX_DIM, 0, legate::Memory::SYSTEM_MEM, 1, legate::Type::Code::FLOAT64),
    std::make_tuple(1, 0, legate::Memory::SYSTEM_MEM, 1, legate::Type::Code::COMPLEX64),
    std::make_tuple(
      2, ALLOCATE_BYTES, legate::Memory::SYSTEM_MEM, 1, legate::Type::Code::COMPLEX128)));

// TODO(joyshennv): issue #1189
// legate::Memory::REGDMA_MEM, legate::Memory::GLOBAL_MEM,
//  legate::Memory::SOCKET_MEM, legate::Memory::Z_COPY_MEM,
//  legate::Memory::GPU_FB_MEM, legate::Memory::DISK_MEM,
//  legate::Memory::HDF_MEM, legate::Memory::FILE_MEM,
//  legate::Memory::LEVEL3_CACHE, legate::Memory::LEVEL2_CACHE,
//  legate::Memory::LEVEL1_CACHE, legate::Memory::GPU_MANAGED_MEM,
//  legate::Memory::GPU_DYNAMIC_MEM));

void test_buffer(std::int32_t dim,
                 std::uint64_t bytes,
                 legate::Memory::Kind kind,
                 std::size_t alignment,
                 legate::Type::Code code)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, BufferTask::TASK_CONFIG.task_id());

  task.add_scalar_arg(legate::Scalar{dim});
  task.add_scalar_arg(legate::Scalar{bytes});
  task.add_scalar_arg(legate::Scalar{kind});
  task.add_scalar_arg(legate::Scalar{alignment});
  task.add_scalar_arg(legate::Scalar{code});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(BufferTaskTest, CreateBuffer)
{
  const auto [dim, bytes, memtype, alignment, code] = GetParam();

  test_buffer(dim, bytes, memtype, alignment, code);
}

TEST_F(BufferUnit, Size)
{
  ASSERT_NO_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
    legate::Point<1>{-1}, legate::Memory::SYSTEM_MEM, alignof(std::max_align_t))));
  ASSERT_NO_THROW(
    static_cast<void>(legate::create_buffer<std::uint64_t>(0, legate::Memory::SYSTEM_MEM)));
}

TEST_F(BufferUnit, OverAlignment)
{
  ASSERT_NO_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
    ALLOCATE_BYTES, legate::Memory::SYSTEM_MEM, OVER_ALIGNMENT)));
}

TEST_F(BufferUnit, BadAlignment)
{
  ASSERT_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
                 ALLOCATE_BYTES, legate::Memory::SYSTEM_MEM, -1)),
               std::domain_error);
  ASSERT_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
                 ALLOCATE_BYTES, legate::Memory::SYSTEM_MEM, 3)),
               std::domain_error);
  ASSERT_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
                 ALLOCATE_BYTES, legate::Memory::SYSTEM_MEM, 0)),
               std::domain_error);
}

}  // namespace buffer_test
