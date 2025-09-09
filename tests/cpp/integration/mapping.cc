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

auto LIBRARY_NAME = "legate.mapping.integration";

// A tiny task that does nothing; we only need mapping to run
struct DummyTask : public legate::LegateTask<DummyTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

// A reduction task to exercise reduction instance mapping paths
struct ReduceTask : public legate::LegateTask<ReduceTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(legate::TaskSignature{}.redops(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext /*context*/) {}
#endif
};

// CPU-only reduction task (registers only CPU variant) so it always runs on CPU
struct ReduceTaskCPUOnly : public legate::LegateTask<ReduceTaskCPUOnly> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(legate::TaskSignature{}.redops(1));

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

}  // namespace

// A simple write task to exercise tighten retry path
struct WriteTask : public legate::LegateTask<WriteTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(legate::TaskSignature{}.outputs(1));

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

namespace {

// Mapper for write task using C-order (default) to create a large instance
class CpuWriteMapperC final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> v;
    auto st  = task.output(0).data();
    auto pol = legate::mapping::InstanceMappingPolicy{}
                 .with_target(legate::mapping::StoreTarget::SYSMEM)
                 .with_ordering(legate::mapping::DimOrdering::c_order());
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

// Mapper for write task using FORTRAN-order to avoid cache reuse on second map
class CpuWriteMapperF final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>&) override
  {
    std::vector<legate::mapping::StoreMapping> v;
    auto st  = task.output(0).data();
    auto pol = legate::mapping::InstanceMappingPolicy{}
                 .with_target(legate::mapping::StoreTarget::SYSMEM)
                 .with_ordering(legate::mapping::DimOrdering::fortran_order());
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

}  // namespace

// Covers BaseMapper::handle_client_mappings duplicate-future-mapping check (L324-L331)
TEST_F(MappingDeathTest, DupFutureStoreMappings)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(
    LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<DupFutureMapper>());
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

// Covers reduction instance creation path (L886-L917, L930-L936) on CPU
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

// Covers reduction cache reuse path on GPU if available (L842-L857 and need_acquire=true)
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

}  // namespace mapping_integration_test
