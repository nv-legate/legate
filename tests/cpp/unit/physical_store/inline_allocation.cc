/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_store_inline_allocation_test {

namespace {

class FutureStoreTask : public legate::LegateTask<FutureStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void FutureStoreTask::cpu_variant(legate::TaskContext context)
{
  auto store = context.output(0).data();

  ASSERT_NE(store.get_inline_allocation().ptr, nullptr);
  ASSERT_EQ(store.get_inline_allocation().strides.size(), store.dim());
  ASSERT_EQ(store.get_inline_allocation().strides.at(0), 0);
  ASSERT_EQ(store.get_inline_allocation().target, legate::mapping::StoreTarget::SYSMEM);
  ASSERT_EQ(store.scalar<std::uint32_t>(), 1);
}

class GPUTask : public legate::LegateTask<GPUTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void gpu_variant(legate::TaskContext context);
};

/*static*/ void GPUTask::gpu_variant(legate::TaskContext context)
{
  const auto store = context.input(0).data();
  const auto alloc = store.get_inline_allocation();

  ASSERT_THAT(
    alloc.target,
    ::testing::AnyOf(legate::mapping::StoreTarget::FBMEM, legate::mapping::StoreTarget::ZCMEM));
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_physical_store_inline_allocation";

  static void registration_callback(legate::Library library)
  {
    FutureStoreTask::register_variants(library);
    GPUTask::register_variants(library);
  }
};

class PhysicalStoreInlineAllocationUnit : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(PhysicalStoreInlineAllocationUnit, FutureStoreByTask)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(Config::LIBRARY_NAME);
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto task          = runtime->create_task(context, FutureStoreTask::TASK_CONFIG.task_id());

  task.add_output(logical_store);
  runtime->submit(std::move(task));
}

TEST_F(PhysicalStoreInlineAllocationUnit, GPUTask)
{
  const auto machine = legate::get_machine().only(legate::mapping::TaskTarget::GPU);

  if (machine.empty()) {
    GTEST_SKIP() << "No GPUs available for GPU task";
  }

  const auto scope    = legate::Scope{machine};
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  const auto store    = runtime->create_store(legate::Shape{2, 2, 2}, legate::int32());
  auto task           = runtime->create_task(lib, GPUTask::TASK_CONFIG.task_id());

  runtime->issue_fill(store, legate::Scalar{std::int32_t{1}});
  task.add_input(store);
  runtime->submit(std::move(task));
}

TEST_F(PhysicalStoreInlineAllocationUnit, ReadOnlyFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto store         = logical_store.get_physical_store();
  auto inline_alloc  = store.get_inline_allocation();

  ASSERT_NE(inline_alloc.ptr, nullptr);
  ASSERT_EQ(inline_alloc.strides.size(), store.dim());
  ASSERT_EQ(inline_alloc.strides.at(0), 0);
  ASSERT_EQ(inline_alloc.target, legate::mapping::StoreTarget::SYSMEM);
}

TEST_F(PhysicalStoreInlineAllocationUnit, TransformedFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto promoted      = logical_store.promote(0, 1);
  auto store         = promoted.get_physical_store();
  auto inline_alloc  = store.get_inline_allocation();

  ASSERT_TRUE(store.transformed());
  ASSERT_NE(inline_alloc.ptr, nullptr);
  ASSERT_EQ(inline_alloc.strides.size(), store.dim());
  ASSERT_EQ(inline_alloc.strides.at(0), 0);
  ASSERT_EQ(inline_alloc.target, legate::mapping::StoreTarget::SYSMEM);
}

TEST_F(PhysicalStoreInlineAllocationUnit, BoundStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2, 3}, legate::uint32());
  auto store         = logical_store.get_physical_store();
  auto inline_alloc  = store.get_inline_allocation();

  ASSERT_NE(inline_alloc.ptr, nullptr);
  ASSERT_EQ(inline_alloc.strides.size(), store.dim());
  ASSERT_EQ(inline_alloc.strides.at(0), sizeof(std::uint32_t) * 3);
  ASSERT_EQ(inline_alloc.strides.at(1), sizeof(std::uint32_t));
  ASSERT_EQ(inline_alloc.target, legate::mapping::StoreTarget::SYSMEM);
}

}  // namespace physical_store_inline_allocation_test
