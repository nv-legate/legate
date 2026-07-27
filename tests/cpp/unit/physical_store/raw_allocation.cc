/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_store_raw_allocation_test {

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
  auto store = context.output(0);

  ASSERT_NE(store.get_buffer().allocation().ptr(), nullptr);
  ASSERT_EQ(store.strides().size(), store.dim());
  ASSERT_EQ(store.strides()[0], 0);
  ASSERT_EQ(store.target(), legate::mapping::StoreTarget::SYSMEM);
}

class GPUTask : public legate::LegateTask<GPUTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void gpu_variant(legate::TaskContext context);
};

/*static*/ void GPUTask::gpu_variant(legate::TaskContext context)
{
  const auto store = context.input(0);

  ASSERT_THAT(
    store.target(),
    ::testing::AnyOf(legate::mapping::StoreTarget::FBMEM, legate::mapping::StoreTarget::ZCMEM));
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_physical_store_raw_allocation";

  static void registration_callback(legate::Library library)
  {
    FutureStoreTask::register_variants(library);
    GPUTask::register_variants(library);
  }
};

class PhysicalStoreRawAllocationUnit : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(PhysicalStoreRawAllocationUnit, FutureStoreByTask)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(Config::LIBRARY_NAME);
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto task          = runtime->create_task(context, FutureStoreTask::TASK_CONFIG.task_id());

  task.add_output(logical_store);
  runtime->submit(std::move(task));
}

TEST_F(PhysicalStoreRawAllocationUnit, GPUTask)
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

TEST_F(PhysicalStoreRawAllocationUnit, ReadOnlyFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto store         = logical_store.get_physical_store();

  ASSERT_NE(store.get_buffer().allocation().ptr(), nullptr);
  ASSERT_EQ(store.strides().size(), store.dim());
  ASSERT_EQ(store.strides()[0], 0);
  ASSERT_EQ(store.target(), legate::mapping::StoreTarget::SYSMEM);
}

TEST_F(PhysicalStoreRawAllocationUnit, TransformedFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1}, legate::Shape{1});
  auto promoted      = logical_store.promote(/*extra_dim=*/0, /*dim_size=*/1);
  auto store         = promoted.get_physical_store();

  ASSERT_NE(store.get_buffer().allocation().ptr(), nullptr);
  ASSERT_EQ(store.strides().size(), store.dim());
  ASSERT_EQ(store.strides()[0], 0);
  ASSERT_EQ(store.target(), legate::mapping::StoreTarget::SYSMEM);
}

TEST_F(PhysicalStoreRawAllocationUnit, BoundStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2, 3}, legate::uint32());
  auto store         = logical_store.get_physical_store();

  // `strides()` reports element strides (not bytes); the (2, 3) row-major store has strides (3, 1).
  ASSERT_NE(store.get_buffer().allocation().ptr(), nullptr);
  ASSERT_EQ(store.strides().size(), store.dim());
  ASSERT_EQ(store.strides()[0], 3);
  ASSERT_EQ(store.strides()[1], 1);
  ASSERT_EQ(store.target(), legate::mapping::StoreTarget::SYSMEM);
}

}  // namespace physical_store_raw_allocation_test
