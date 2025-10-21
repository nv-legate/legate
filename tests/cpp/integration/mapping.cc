/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/mapping.h>

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace mapping_integration_test {

namespace {

using MappingIntegrationTests = DefaultFixture;
using MappingDeathTest        = MappingIntegrationTests;

// A tiny task that does nothing; we only need mapping to run
class DummyTask : public legate::LegateTask<DummyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

// A reduction task to exercise reduction instance mapping paths
class ReduceTask : public legate::LegateTask<ReduceTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(legate::TaskSignature{}.redops(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext /*context*/) {}
#endif
};

// CPU-only reduction task (registers only CPU variant) so it always runs on CPU
class ReduceTaskCPUOnly : public legate::LegateTask<ReduceTaskCPUOnly> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(legate::TaskSignature{}.redops(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

// Task with multiple stores, used to test partial mapping failure scenarios
class MultiStoreTask : public legate::LegateTask<MultiStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(
      legate::TaskSignature{}.inputs(1).redops(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

// A custom mapper that returns duplicate mappings for the same future-backed store
class DupFutureMapper final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> mappings;
    const auto store = task.input(0).data();

    // Two different policies for the SAME future store -> triggers duplicate detection
    auto p1 =
      legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::SYSMEM);
    auto p2 =
      legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::ZCMEM);

    mappings.emplace_back(legate::mapping::StoreMapping::create(store, std::move(p1)));
    mappings.emplace_back(legate::mapping::StoreMapping::create(store, std::move(p2)));
    return mappings;
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task&,
                                                  legate::mapping::StoreTarget) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID) override { return {}; }
};

// CPU mapper (maps to host memory) - deterministic coverage of creation path
class CpuRedMapper final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> v;
    auto st = task.reduction(0).data();
    auto pol =
      legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::SYSMEM);
    v.emplace_back(legate::mapping::StoreMapping::create(st, std::move(pol)));
    return v;
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task&,
                                                  legate::mapping::StoreTarget) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID) override { return {}; }
};

// GPU mapper (maps to FB) - enables cache reuse path if GPUs exist
class GpuRedMapper final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> v;
    auto st = task.reduction(0).data();
    auto pol =
      legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::FBMEM);
    v.emplace_back(legate::mapping::StoreMapping::create(st, std::move(pol)));
    return v;
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task&,
                                                  legate::mapping::StoreTarget) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID) override { return {}; }
};

// Custom mapper: first store uses small exact policy, second store uses very large non-exact policy
class PartialFailMapper final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> mappings;

    // First store (input): use exact=true to ensure successful mapping
    if (!task.inputs().empty()) {
      auto st  = task.input(0).data();
      auto pol = legate::mapping::InstanceMappingPolicy{}
                   .with_target(legate::mapping::StoreTarget::SYSMEM)
                   .with_exact(true);
      mappings.emplace_back(legate::mapping::StoreMapping::create(st, std::move(pol)));
    }

    // Second store (reduction): use exact=false, may fail
    if (!task.reductions().empty()) {
      auto st  = task.reduction(0).data();
      auto pol = legate::mapping::InstanceMappingPolicy{}
                   .with_target(legate::mapping::StoreTarget::SYSMEM)
                   .with_exact(false);
      mappings.emplace_back(legate::mapping::StoreMapping::create(st, std::move(pol)));
    }

    return mappings;
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task&,
                                                  legate::mapping::StoreTarget) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID) override { return {}; }
};

// Custom mapper: use exact=false for read-only inputs
class ReadOnlyNonExactMapper final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> mappings;

    // Use exact=false for input store
    if (!task.inputs().empty()) {
      auto st  = task.input(0).data();
      auto pol = legate::mapping::InstanceMappingPolicy{}
                   .with_target(legate::mapping::StoreTarget::SYSMEM)
                   .with_exact(false);
      mappings.emplace_back(legate::mapping::StoreMapping::create(st, std::move(pol)));
    }

    return mappings;
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task&,
                                                  legate::mapping::StoreTarget) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID) override { return {}; }
};

}  // namespace

// Covers BaseMapper::handle_client_mappings duplicate-future-mapping check
TEST_F(MappingDeathTest, DupFutureStoreMappings)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library("legate.mapping.dup_future_store_mapping",
                                         legate::ResourceConfig{},
                                         std::make_unique<DupFutureMapper>());
  DummyTask::register_variants(library);
  constexpr auto scalar_value = 42;
  auto scalar_store           = runtime->create_store(legate::Scalar{scalar_value});

  // Launch the task with the future-backed store as input; mapping should abort due to duplicates
  ASSERT_DEATH(
    {
      auto task = runtime->create_task(
        library, DummyTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{1});
      task.add_input(scalar_store);
      runtime->submit(std::move(task));
      // Ensure mapping proceeds
      runtime->issue_execution_fence(true);
    },
    "returned duplicate store mappings");
}

// Covers reduction instance creation path on CPU
TEST_F(MappingIntegrationTests, ReductionCreateCPU)
{
  auto* rt = legate::Runtime::get_runtime();
  auto lib = rt->create_library(
    "legate.mapping.reduction.cpu", legate::ResourceConfig{}, std::make_unique<CpuRedMapper>());
  ReduceTaskCPUOnly::register_variants(lib);

  // Single region reduction store
  constexpr auto shape_value = 16;
  auto store                 = rt->create_store(legate::Shape{shape_value}, legate::int32());
  rt->issue_fill(store, legate::Scalar{1});

  auto t =
    rt->create_task(lib, ReduceTaskCPUOnly::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{1});
  t.add_reduction(store, legate::ReductionOpKind::ADD);
  rt->submit(std::move(t));
  rt->issue_execution_fence(true);
}

// Covers reduction cache reuse path on GPU if available
TEST_F(MappingIntegrationTests, ReductionReuseGPU)
{
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP();
  }

  auto* rt = legate::Runtime::get_runtime();
  auto lib = rt->create_library(
    "legate.mapping.reduction.gpu", legate::ResourceConfig{}, std::make_unique<GpuRedMapper>());
  ReduceTask::register_variants(lib);

  constexpr auto shape_value = 32;
  auto store                 = rt->create_store(legate::Shape{shape_value}, legate::int32());
  rt->issue_fill(store, legate::Scalar{2});

  // First submit: create reduction instance
  {
    auto t1 =
      rt->create_task(lib, ReduceTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{1});
    t1.add_reduction(store, legate::ReductionOpKind::ADD);
    rt->submit(std::move(t1));
    rt->issue_execution_fence(true);
  }
  // Second submit: reuse cached reduction instance (acquire only)
  {
    auto t2 =
      rt->create_task(lib, ReduceTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{1});
    t2.add_reduction(store, legate::ReductionOpKind::ADD);
    rt->submit(std::move(t2));
    rt->issue_execution_fence(true);
  }
}

TEST_F(MappingDeathTest, MapSingleStoreFail)
{
  auto* rt = legate::Runtime::get_runtime();
  auto lib = rt->create_library("legate.mapping.single_store_mapping_fail",
                                legate::ResourceConfig{},
                                std::make_unique<CpuRedMapper>());
  ReduceTaskCPUOnly::register_variants(lib);

  constexpr std::uint64_t huge_size = 1ULL << 40;  // 1TB
  auto store                        = rt->create_store(legate::Shape{huge_size}, legate::int32());
  rt->issue_fill(store, legate::Scalar{0});

  const legate::tuple<std::uint64_t> launch{1};
  auto task = rt->create_task(lib, ReduceTaskCPUOnly::TASK_CONFIG.task_id(), launch);
  task.add_reduction(store, legate::ReductionOpKind::ADD);
  ASSERT_DEATH(
    {
      rt->submit(std::move(task));
      rt->issue_execution_fence(true);
    },
    "Out of memory");
}

TEST_F(MappingDeathTest, MapMultipleStoresPartialFail)
{
  auto* rt = legate::Runtime::get_runtime();
  auto lib = rt->create_library("legate.mapping.multiple_stores_mapping_fail",
                                legate::ResourceConfig{},
                                std::make_unique<PartialFailMapper>());
  MultiStoreTask::register_variants(lib);

  // Create a small input store (will succeed in mapping)
  constexpr std::uint64_t SHAPE_SIZE = 1000;
  auto input_store                   = rt->create_store(legate::Shape{SHAPE_SIZE}, legate::int32());
  rt->issue_fill(input_store, legate::Scalar{0});

  // Create a very large reduction store (will fail in mapping)
  constexpr std::uint64_t huge_size = 1ULL << 40;  // 1TB
  auto reduction_store              = rt->create_store(legate::Shape{huge_size}, legate::int32());
  rt->issue_fill(reduction_store, legate::Scalar{0});

  const legate::tuple<std::uint64_t> launch{1};
  auto task = rt->create_task(lib, MultiStoreTask::TASK_CONFIG.task_id(), launch);
  task.add_input(input_store);
  task.add_reduction(reduction_store, legate::ReductionOpKind::ADD);
  ASSERT_DEATH(
    {
      rt->submit(std::move(task));
      rt->issue_execution_fence(true);
    },
    "Out of memory");  // Expected output contains "Out of memory"
}

// Use read-only input + exact=false, read-only store will be skipped during tighten
TEST_F(MappingDeathTest, MapLegateStoresReadOnlyNonExact)
{
  auto* rt = legate::Runtime::get_runtime();
  auto lib = rt->create_library("legate.mapping.readonly_nonexact",
                                legate::ResourceConfig{},
                                std::make_unique<ReadOnlyNonExactMapper>());
  DummyTask::register_variants(lib);

  // Create a very large read-only store
  constexpr std::uint64_t huge_size = 1ULL << 40;  // 1TB
  auto store                        = rt->create_store(legate::Shape{huge_size}, legate::int32());
  rt->issue_fill(store, legate::Scalar{0});

  const legate::tuple<std::uint64_t> launch{1};
  auto task = rt->create_task(lib, DummyTask::TASK_CONFIG.task_id(), launch);
  task.add_input(store);  // Read-only input
  ASSERT_DEATH(
    {
      rt->submit(std::move(task));
      rt->issue_execution_fence(true);
    },
    "Out of memory");
}

}  // namespace mapping_integration_test
