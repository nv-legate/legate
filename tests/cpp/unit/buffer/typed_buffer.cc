/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <utilities/utilities.h>

namespace buffer_test {

namespace {

constexpr std::uint64_t ALLOCATE_BYTES = 10;
constexpr std::uint64_t OVER_ALIGNMENT = 128;

// An optional store target: std::nullopt means "let the runtime pick based on the executing
// processor".
using OptStoreTarget = std::optional<legate::mapping::StoreTarget>;

// A store target has to cross the task boundary as a scalar. There is no "none" `StoreTarget`, so
// encode "pick automatically" as -1 and any concrete target as its underlying value.
[[nodiscard]] std::int32_t encode_store_target(OptStoreTarget target)
{
  return target.has_value() ? static_cast<std::int32_t>(*target) : -1;
}

[[nodiscard]] OptStoreTarget decode_store_target(std::int32_t value)
{
  if (value < 0) {
    return std::nullopt;
  }
  return static_cast<legate::mapping::StoreTarget>(value);
}

class BufferFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(std::uint64_t bytes, OptStoreTarget target, std::uint64_t alignment) const
  {
    using T = legate::type_of_t<CODE>;

    // An unspecified target on a cpu is always mapped to SYSMEM
    const auto expected = target.value_or(legate::mapping::StoreTarget::SYSMEM);

    switch (DIM) {
      case 1: {
        auto buffer = legate::create_buffer<T>(bytes, target, alignment);
        EXPECT_EQ(buffer.buffer().target(), expected);
        break;
      }
      default: {
        auto buffer = legate::create_buffer<T, DIM>(
          legate::NDPoint<DIM>{static_cast<legate::coord_t>(bytes)}, target, alignment);
        EXPECT_EQ(buffer.buffer().target(), expected);
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
  auto target    = decode_store_target(context.scalar(2).value<std::int32_t>());
  auto alignment = context.scalar(3).value<std::size_t>();
  auto code      = context.scalar(4).value<legate::Type::Code>();

  legate::double_dispatch(dim, code, BufferFn{}, bytes, target, alignment);
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
      std::tuple<int, std::size_t, OptStoreTarget, std::size_t, legate::Type::Code>> {};

INSTANTIATE_TEST_SUITE_P(
  BufferUnit,
  BufferTaskTest,
  ::testing::Values(
    std::make_tuple(1, 0, OptStoreTarget{}, 1, legate::Type::Code::BOOL),
    std::make_tuple(
      2, ALLOCATE_BYTES, OptStoreTarget{}, alignof(std::max_align_t), legate::Type::Code::INT8),
    std::make_tuple(
      3, ALLOCATE_BYTES, OptStoreTarget{}, alignof(std::max_align_t), legate::Type::Code::INT16),
    std::make_tuple(LEGATE_MAX_DIM, 0, OptStoreTarget{}, 1, legate::Type::Code::INT32),
    std::make_tuple(1, 0, OptStoreTarget{}, 1, legate::Type::Code::INT64),
    std::make_tuple(
      2, ALLOCATE_BYTES, OptStoreTarget{}, alignof(std::max_align_t), legate::Type::Code::UINT8),
    std::make_tuple(
      3, ALLOCATE_BYTES, OptStoreTarget{}, alignof(std::max_align_t), legate::Type::Code::UINT16),
    std::make_tuple(LEGATE_MAX_DIM,
                    0,
                    OptStoreTarget{legate::mapping::StoreTarget::SYSMEM},
                    1,
                    legate::Type::Code::UINT32),
    std::make_tuple(
      1, 0, OptStoreTarget{legate::mapping::StoreTarget::SYSMEM}, 1, legate::Type::Code::UINT64),
    std::make_tuple(2,
                    ALLOCATE_BYTES,
                    OptStoreTarget{legate::mapping::StoreTarget::SYSMEM},
                    alignof(std::max_align_t),
                    legate::Type::Code::FLOAT16),
    std::make_tuple(3,
                    ALLOCATE_BYTES,
                    OptStoreTarget{legate::mapping::StoreTarget::SYSMEM},
                    alignof(std::max_align_t),
                    legate::Type::Code::FLOAT32),
    std::make_tuple(LEGATE_MAX_DIM,
                    0,
                    OptStoreTarget{legate::mapping::StoreTarget::SYSMEM},
                    1,
                    legate::Type::Code::FLOAT64),
    std::make_tuple(
      1, 0, OptStoreTarget{legate::mapping::StoreTarget::SYSMEM}, 1, legate::Type::Code::COMPLEX64),
    std::make_tuple(2,
                    ALLOCATE_BYTES,
                    OptStoreTarget{legate::mapping::StoreTarget::SYSMEM},
                    1,
                    legate::Type::Code::COMPLEX128)));

// TODO(joyshennv): issue #1189
// Also exercise the other store targets:
//  legate::mapping::StoreTarget::SOCKETMEM, legate::mapping::StoreTarget::ZCMEM,
//  legate::mapping::StoreTarget::FBMEM

void test_buffer(std::int32_t dim,
                 std::uint64_t bytes,
                 OptStoreTarget target,
                 std::size_t alignment,
                 legate::Type::Code code)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, BufferTask::TASK_CONFIG.task_id());

  task.add_scalar_arg(legate::Scalar{dim});
  task.add_scalar_arg(legate::Scalar{bytes});
  task.add_scalar_arg(legate::Scalar{encode_store_target(target)});
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
    legate::NDPoint<1>{-1}, legate::mapping::StoreTarget::SYSMEM, alignof(std::max_align_t))));
  ASSERT_NO_THROW(static_cast<void>(
    legate::create_buffer<std::uint64_t>(0, legate::mapping::StoreTarget::SYSMEM)));
}

TEST_F(BufferUnit, OverAlignment)
{
  ASSERT_NO_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
    ALLOCATE_BYTES, legate::mapping::StoreTarget::SYSMEM, OVER_ALIGNMENT)));
}

TEST_F(BufferUnit, BadAlignment)
{
  ASSERT_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
                 ALLOCATE_BYTES, legate::mapping::StoreTarget::SYSMEM, -1)),
               std::domain_error);
  ASSERT_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
                 ALLOCATE_BYTES, legate::mapping::StoreTarget::SYSMEM, 3)),
               std::domain_error);
  ASSERT_THROW(static_cast<void>(legate::create_buffer<std::uint64_t>(
                 ALLOCATE_BYTES, legate::mapping::StoreTarget::SYSMEM, 0)),
               std::domain_error);
}

}  // namespace buffer_test
