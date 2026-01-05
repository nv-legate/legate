/*A
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/tuple.h>

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

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "legate.task_local_buffer";

  static void registration_callback(legate::Library library)
  {
    BasicBufferTask::register_variants(library);
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

}  // namespace task_local_buffer_test
