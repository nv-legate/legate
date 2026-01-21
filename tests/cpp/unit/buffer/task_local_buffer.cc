/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/tuple.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace task_local_buffer_test {

class BasicBufferTask : public legate::LegateTask<BasicBufferTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true))
      .with_signature(legate::TaskSignature{}.scalars(2));

  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
 private:
  // We would like to parameterize the test over the mem-kinds as well, but then if you get it
  // wrong Legion simply aborts with:
  //
  // LEGION ERROR: Unable to find associated Visible to all processors on a node memory for
  // Throughput core processor when performing an DeferredBuffer creation in task
  // task_local_buffer_test::BasicBufferTask
  //
  // (This error indicates that we tried to allocate SYSMEM on a GPU task). So instead of going
  // through complex hoops to exclude certain memkinds based on which variant we think will get
  // executed, we just mandate the memory based on the task.
  static void common_variant_(legate::TaskContext context, legate::mapping::StoreTarget mem_kind);
};

/* static */ void BasicBufferTask::common_variant_(legate::TaskContext context,
                                                   legate::mapping::StoreTarget mem_kind)
{
  const auto code   = context.scalar(0).value<legate::Type::Code>();
  const auto bounds = context.scalar(1).values<std::uint64_t>();
  const auto ty     = legate::primitive_type(code);
  const auto buf    = legate::TaskLocalBuffer{ty, bounds, mem_kind};

  ASSERT_EQ(buf.type(), ty);
  ASSERT_EQ(buf.dim(), bounds.size());
  ASSERT_EQ(buf.domain(), legate::detail::to_domain(bounds));
  ASSERT_EQ(buf.memory_kind(), mem_kind);
}

/* static */ void BasicBufferTask::cpu_variant(legate::TaskContext context)
{
  common_variant_(context, legate::mapping::StoreTarget::SYSMEM);
}

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
/* static */ void BasicBufferTask::gpu_variant(legate::TaskContext context)
{
  common_variant_(context, legate::mapping::StoreTarget::FBMEM);
}
#endif

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
/* static */ void BasicBufferTask::omp_variant(legate::TaskContext context)
{
  common_variant_(context, legate::mapping::StoreTarget::SOCKETMEM);
}
#endif

class CheckTypeSizeMismatchTask : public legate::LegateTask<CheckTypeSizeMismatchTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    // Size mismatch: Buffer<std::int32_t> (size=4) vs int64 type (size=8)
    auto buf = legate::create_buffer<std::int32_t>(/*size=*/9, legate::Memory::SYSTEM_MEM);
    ASSERT_THAT(
      [&] {
        static_cast<void>(
          legate::TaskLocalBuffer{buf, legate::primitive_type(legate::Type::Code::INT64)});
      },
      ::testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("incompatible type sizes")));
  }
};

class CheckTypeAlignmentMismatchTask : public legate::LegateTask<CheckTypeAlignmentMismatchTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    // Alignment mismatch: directly call check_type with matching size but mismatched alignment
    // Use int32 type (size=4, align=4) but pass align_of=8 to trigger alignment check branch
    const auto ty = legate::primitive_type(legate::Type::Code::INT32);
    ASSERT_THAT(
      [&] { legate::untyped_buffer_detail::check_type(ty, ty.size(), ty.alignment() * 2); },
      ::testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("incompatible type alignment")));
  }
};

class CheckTypeSuccessTask : public legate::LegateTask<CheckTypeSuccessTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    // Successful case: Buffer<std::int32_t> with int32 type (should not throw)
    auto buf = legate::create_buffer<std::int32_t>(/*size=*/8, legate::Memory::SYSTEM_MEM);
    const legate::TaskLocalBuffer task_local_buf{buf,
                                                 legate::primitive_type(legate::Type::Code::INT32)};

    auto converted_buf = static_cast<legate::Buffer<std::int32_t>>(task_local_buf);
    ASSERT_EQ(converted_buf.ptr(0), buf.ptr(0));

    auto buf_no_memkind =
      legate::create_buffer<std::int32_t>(/*size=*/7);  // Uses NO_MEMKIND by default
    ASSERT_NE(buf_no_memkind.ptr(0), nullptr);
  }
};

class CopyConstructorTask : public legate::LegateTask<CopyConstructorTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{4}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    const auto ty     = legate::primitive_type(legate::Type::Code::INT32);
    const auto bounds = std::vector<std::uint64_t>{10};
    const auto orig   = legate::TaskLocalBuffer{ty, bounds, legate::mapping::StoreTarget::SYSMEM};
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization) - testing copy ctor
    const auto copy = orig;

    ASSERT_EQ(copy.type(), orig.type());
    ASSERT_EQ(copy.dim(), orig.dim());
    ASSERT_EQ(copy.domain(), orig.domain());
    ASSERT_EQ(copy.memory_kind(), orig.memory_kind());
  }
};

class MoveConstructorTask : public legate::LegateTask<MoveConstructorTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{5}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    const auto ty     = legate::primitive_type(legate::Type::Code::INT64);
    const auto bounds = std::vector<std::uint64_t>{20};
    auto orig         = legate::TaskLocalBuffer{ty, bounds, legate::mapping::StoreTarget::SYSMEM};
    const auto moved  = std::move(orig);  // Move constructor

    ASSERT_EQ(moved.type(), ty);
    ASSERT_EQ(moved.dim(), 1);
    ASSERT_EQ(moved.memory_kind(), legate::mapping::StoreTarget::SYSMEM);
  }
};

class DefaultMemKindTask : public legate::LegateTask<DefaultMemKindTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{6}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    // Test TaskLocalBuffer without mem_kind - triggers find_memory_kind_for_executing_processor
    const auto ty     = legate::primitive_type(legate::Type::Code::INT32);
    const auto bounds = std::vector<std::uint64_t>{10};
    // Note: not passing mem_kind, so it defaults to std::nullopt
    const auto buf = legate::TaskLocalBuffer{ty, bounds};

    ASSERT_EQ(buf.type(), ty);
    ASSERT_EQ(buf.dim(), 1);
    // Memory kind is determined by the executing processor (CPU task -> SYSMEM)
    ASSERT_EQ(buf.memory_kind(), legate::mapping::StoreTarget::SYSMEM);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "legate.task_local_buffer";

  static void registration_callback(legate::Library library)
  {
    BasicBufferTask::register_variants(library);
    CheckTypeSizeMismatchTask::register_variants(library);
    CheckTypeAlignmentMismatchTask::register_variants(library);
    CheckTypeSuccessTask::register_variants(library);
    CopyConstructorTask::register_variants(library);
    MoveConstructorTask::register_variants(library);
    DefaultMemKindTask::register_variants(library);
  }
};

class TaskLocalBufferUnit : public RegisterOnceFixture<Config>,
                            public ::testing::WithParamInterface<
                              std::tuple<legate::Type::Code, std::vector<std::uint64_t>>> {};

INSTANTIATE_TEST_SUITE_P(,
                         TaskLocalBufferUnit,
                         // Tests a selection of types, mostly just to check sizes.
                         ::testing::Combine(::testing::Values(legate::Type::Code::INT8,
                                                              legate::Type::Code::UINT16,
                                                              legate::Type::Code::INT32,
                                                              legate::Type::Code::UINT64,
                                                              legate::Type::Code::FLOAT32,
                                                              legate::Type::Code::FLOAT64,
                                                              legate::Type::Code::COMPLEX64,
                                                              legate::Type::Code::COMPLEX128),
                                            ::testing::Values(std::vector<std::uint64_t>{1},
                                                              std::vector<std::uint64_t>{1, 2})));

TEST_P(TaskLocalBufferUnit, Basic)
{
  auto* const runtime        = legate::Runtime::get_runtime();
  const auto lib             = runtime->find_library(Config::LIBRARY_NAME);
  auto task                  = runtime->create_task(lib, BasicBufferTask::TASK_CONFIG.task_id());
  const auto [code, extents] = GetParam();

  task.add_scalar_arg(legate::Scalar{code});
  task.add_scalar_arg(legate::Scalar{extents});
  runtime->submit(std::move(task));
}

class CheckTypeUnit : public RegisterOnceFixture<Config> {};

TEST_F(CheckTypeUnit, SizeMismatch)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, CheckTypeSizeMismatchTask::TASK_CONFIG.task_id());

  runtime->submit(std::move(task));
}

TEST_F(CheckTypeUnit, AlignmentMismatch)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task = runtime->create_task(lib, CheckTypeAlignmentMismatchTask::TASK_CONFIG.task_id());

  runtime->submit(std::move(task));
}

TEST_F(CheckTypeUnit, Success)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, CheckTypeSuccessTask::TASK_CONFIG.task_id());

  runtime->submit(std::move(task));
}

class ConstructorUnit : public RegisterOnceFixture<Config> {};

TEST_F(ConstructorUnit, CopyConstructor)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, CopyConstructorTask::TASK_CONFIG.task_id());

  runtime->submit(std::move(task));
}

TEST_F(ConstructorUnit, MoveConstructor)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, MoveConstructorTask::TASK_CONFIG.task_id());

  runtime->submit(std::move(task));
}

TEST_F(ConstructorUnit, DefaultMemKind)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, DefaultMemKindTask::TASK_CONFIG.task_id());

  runtime->submit(std::move(task));
}

}  // namespace task_local_buffer_test
