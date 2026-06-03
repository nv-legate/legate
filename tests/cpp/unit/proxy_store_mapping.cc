/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/proxy_store_mapping.h>

#include <legate.h>

#include <legate/mapping/detail/proxy_store_mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/task.h>
#include <legate/partitioning/proxy.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utilities/utilities.h>

namespace proxy_store_mapping_test {

namespace {

constexpr auto DEFAULT_TARGET = legate::mapping::StoreTarget::SYSMEM;
constexpr auto FILL_VALUE     = std::int32_t{1};
constexpr auto INLINE_INPUTS  = std::size_t{2};
constexpr auto INLINE_OUTPUTS = std::size_t{2};

using InstanceMappingPolicy = legate::mapping::InstanceMappingPolicy;
using PolicyVector          = legate::detail::SmallVector<InstanceMappingPolicy>;

class CheckTaskTargetTask : public legate::LegateTask<CheckTaskTargetTask> {
 public:
  static inline const auto
    TASK_CONFIG =  // NOLINT(cert-err58-cpp,bugprone-throwing-static-initialization)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_store_mappings(
      {{{legate::proxy::inputs, legate::mapping::StoreTarget::SYSMEM},
        {legate::proxy::outputs[0], legate::mapping::StoreTarget::SYSMEM}}});

  static void cpu_variant(legate::TaskContext ctx) { task_body_(ctx); }

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext ctx) { task_body_(ctx); }
#endif

 private:
  // The real test here is if the OpenMP variants also use sysmem
  static void task_body_(legate::TaskContext ctx)
  {
    for (std::uint32_t i = 0; i < ctx.num_inputs(); ++i) {
      const auto target = ctx.input(i).target();

      ASSERT_EQ(target, legate::mapping::StoreTarget::SYSMEM);
    }

    {
      ASSERT_EQ(ctx.num_outputs(), 1);

      const auto target = ctx.output(0).target();

      ASSERT_EQ(target, legate::mapping::StoreTarget::SYSMEM);
    }
  }
};

class CheckDefaultOutputTargetTask : public legate::LegateTask<CheckDefaultOutputTargetTask> {
 public:
  static inline const auto
    TASK_CONFIG =  // NOLINT(cert-err58-cpp,bugprone-throwing-static-initialization)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_store_mappings(
      {{legate::proxy::outputs, std::nullopt}});

  static void cpu_variant(legate::TaskContext ctx)
  {
    for (std::uint32_t i = 0; i < ctx.num_outputs(); ++i) {
      ASSERT_EQ(ctx.output(i).target(), DEFAULT_TARGET);
    }
  }
};

class CheckIndexedInputTargetTask : public legate::LegateTask<CheckIndexedInputTargetTask> {
 public:
  static inline const auto
    TASK_CONFIG =  // NOLINT(cert-err58-cpp,bugprone-throwing-static-initialization)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_store_mappings(
      {{{legate::proxy::inputs[0], DEFAULT_TARGET}}});

  static void cpu_variant(legate::TaskContext ctx)
  {
    ASSERT_EQ(ctx.num_inputs(), 1);
    ASSERT_EQ(ctx.input(0).target(), DEFAULT_TARGET);
  }
};

class CheckAllReductionTargetTask : public legate::LegateTask<CheckAllReductionTargetTask> {
 public:
  static inline const auto
    TASK_CONFIG =  // NOLINT(cert-err58-cpp,bugprone-throwing-static-initialization)
    legate::TaskConfig{legate::LocalTaskID{3}}.with_store_mappings(
      {{{legate::proxy::reductions, DEFAULT_TARGET}}});

  static void cpu_variant(legate::TaskContext ctx)
  {
    ASSERT_EQ(ctx.num_reductions(), 1);
    ASSERT_EQ(ctx.reduction(0).target(), DEFAULT_TARGET);
  }
};

class CheckIndexedReductionTargetTask : public legate::LegateTask<CheckIndexedReductionTargetTask> {
 public:
  static inline const auto
    TASK_CONFIG =  // NOLINT(cert-err58-cpp,bugprone-throwing-static-initialization)
    legate::TaskConfig{legate::LocalTaskID{4}}.with_store_mappings(
      {{{legate::proxy::reductions[0], DEFAULT_TARGET}}});

  static void cpu_variant(legate::TaskContext ctx)
  {
    ASSERT_EQ(ctx.num_reductions(), 1);
    ASSERT_EQ(ctx.reduction(0).target(), DEFAULT_TARGET);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_proxy_store_mapping";

  static void registration_callback(legate::Library library)
  {
    CheckTaskTargetTask::register_variants(library);
    CheckDefaultOutputTargetTask::register_variants(library);
    CheckIndexedInputTargetTask::register_variants(library);
    CheckAllReductionTargetTask::register_variants(library);
    CheckIndexedReductionTargetTask::register_variants(library);
  }
};

class ProxyStoreMappingUnit : public RegisterOnceFixture<Config> {};

struct InlinePolicyVectors {
  PolicyVector inputs;
  PolicyVector outputs;
  PolicyVector reductions;
};

[[nodiscard]] legate::AutoTask create_task_with_all_inline_arguments()
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, CheckTaskTargetTask::TASK_CONFIG.task_id());

  for (std::size_t i = 0; i < INLINE_INPUTS; ++i) {
    auto store = runtime->create_store(legate::Shape{1, i + 1}, legate::int32());

    runtime->issue_fill(store, legate::Scalar{FILL_VALUE});
    task.add_input(std::move(store));
  }

  for (std::size_t i = 0; i < INLINE_OUTPUTS; ++i) {
    auto store = runtime->create_store(legate::Shape{1, i + 1}, legate::int32());

    task.add_output(std::move(store));
  }

  auto reduction = runtime->create_store(legate::Shape{1, 1}, legate::int32());

  runtime->issue_fill(reduction, legate::Scalar{FILL_VALUE});
  task.add_reduction(std::move(reduction), legate::ReductionOpKind::ADD);

  return task;
}

[[nodiscard]] InlinePolicyVectors make_inline_policy_vectors(const legate::AutoTask& task)
{
  const auto default_policy = InstanceMappingPolicy{};

  return {
    PolicyVector{legate::detail::tags::size_tag, task.impl_()->inputs().size(), default_policy},
    PolicyVector{legate::detail::tags::size_tag, task.impl_()->outputs().size(), default_policy},
    PolicyVector{
      legate::detail::tags::size_tag, task.impl_()->reductions().size(), default_policy}};
}

[[nodiscard]] legate::mapping::ProxyInstanceMappingPolicy make_proxy_policy()
{
  return legate::mapping::ProxyInstanceMappingPolicy{}
    .with_target(DEFAULT_TARGET)
    .with_exact(true)
    .with_allocation_policy(legate::mapping::AllocPolicy::MUST_ALLOC)
    .with_redundant(false)
    .with_ordering(legate::mapping::DimOrdering::fortran_order());
}

[[nodiscard]] InstanceMappingPolicy make_instance_policy(
  legate::mapping::StoreTarget target, const legate::mapping::ProxyInstanceMappingPolicy& policy)
{
  return InstanceMappingPolicy{
    target, policy.allocation, policy.ordering, policy.exact, policy.redundant};
}

}  // namespace

TEST_F(ProxyStoreMappingUnit, Basic)
{
  constexpr auto target = legate::mapping::StoreTarget::FBMEM;
  const auto psm        = legate::mapping::ProxyStoreMapping{legate::proxy::outputs, target};

  ASSERT_EQ(psm.policy(), legate::mapping::ProxyInstanceMappingPolicy{}.with_target(target));

  auto&& impl = psm.impl();

  ASSERT_THAT(impl->stores(),
              ::testing::VariantWith<legate::ProxyOutputArguments>(legate::ProxyOutputArguments{}));
}

TEST_F(ProxyStoreMappingUnit, FromPolicy)
{
  const auto policy = legate::mapping::ProxyInstanceMappingPolicy{}
                        .with_target(legate::mapping::StoreTarget::SYSMEM)
                        .with_exact(true)
                        .with_allocation_policy(legate::mapping::AllocPolicy::MUST_ALLOC)
                        .with_redundant(false)
                        .with_ordering(legate::mapping::DimOrdering::fortran_order());
  // Need this copy because ProxyStoreMapping takes by rvalue ref
  auto cpy = policy;

  const auto psm = legate::mapping::ProxyStoreMapping{legate::proxy::outputs, std::move(cpy)};

  ASSERT_EQ(psm.policy(), policy);
}

TEST(ProxyInstanceMappingPolicyUnit, NotEqual)
{
  const auto base      = legate::mapping::ProxyInstanceMappingPolicy{}
                           .with_target(legate::mapping::StoreTarget::SYSMEM)
                           .with_exact(true);
  const auto different = legate::mapping::ProxyInstanceMappingPolicy{}
                           .with_target(legate::mapping::StoreTarget::FBMEM)
                           .with_exact(true);

  ASSERT_NE(base, different);
}

TEST_F(ProxyStoreMappingUnit, CompareMappings)
{
  const auto mapping   = legate::mapping::ProxyStoreMapping{legate::proxy::outputs, DEFAULT_TARGET};
  const auto same      = legate::mapping::ProxyStoreMapping{legate::proxy::outputs, DEFAULT_TARGET};
  const auto different = legate::mapping::ProxyStoreMapping{legate::proxy::inputs, DEFAULT_TARGET};

  ASSERT_EQ(*mapping.impl(), *mapping.impl());
  ASSERT_EQ(*mapping.impl(), *same.impl());
  ASSERT_NE(*mapping.impl(), *different.impl());
  ASSERT_NE(mapping, different);
}

TEST_F(ProxyStoreMappingUnit, Apply)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, CheckTaskTargetTask::TASK_CONFIG.task_id());

  for (std::size_t i = 0; i < 2; ++i) {
    auto store = runtime->create_store(legate::Shape{1, i}, legate::int32());

    runtime->issue_fill(store, legate::Scalar{1});
    task.add_input(std::move(store));
  }

  {
    auto store = runtime->create_store(legate::Shape{2, 2}, legate::int32());

    runtime->issue_fill(store, legate::Scalar{1});
    task.add_output(std::move(store));
  }

  runtime->submit(std::move(task));
}

TEST_F(ProxyStoreMappingUnit, ApplyDefaultOutputMapping)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task = runtime->create_task(lib, CheckDefaultOutputTargetTask::TASK_CONFIG.task_id());

  for (std::size_t i = 0; i < 2; ++i) {
    auto store = runtime->create_store(legate::Shape{1, i + 1}, legate::int32());

    task.add_output(std::move(store));
  }

  runtime->submit(std::move(task));
}

TEST_F(ProxyStoreMappingUnit, ApplyIndexedInputMapping)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task  = runtime->create_task(lib, CheckIndexedInputTargetTask::TASK_CONFIG.task_id());
  auto store = runtime->create_store(legate::Shape{1, 1}, legate::int32());

  runtime->issue_fill(store, legate::Scalar{FILL_VALUE});
  task.add_input(std::move(store));
  runtime->submit(std::move(task));
}

TEST_F(ProxyStoreMappingUnit, ApplyAllReductionMapping)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task  = runtime->create_task(lib, CheckAllReductionTargetTask::TASK_CONFIG.task_id());
  auto store = runtime->create_store(legate::Shape{1, 1}, legate::int32());

  runtime->issue_fill(store, legate::Scalar{FILL_VALUE});
  task.add_reduction(std::move(store), legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
}

TEST_F(ProxyStoreMappingUnit, ApplyIndexedReductionMapping)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task  = runtime->create_task(lib, CheckIndexedReductionTargetTask::TASK_CONFIG.task_id());
  auto store = runtime->create_store(legate::Shape{1, 1}, legate::int32());

  runtime->issue_fill(store, legate::Scalar{FILL_VALUE});
  task.add_reduction(std::move(store), legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
}

TEST_F(ProxyStoreMappingUnit, ApplyInline)
{
  auto task           = create_task_with_all_inline_arguments();
  auto policy_vectors = make_inline_policy_vectors(task);
  const auto policy   = make_proxy_policy();
  const auto options  = std::array{legate::mapping::StoreTarget::FBMEM, DEFAULT_TARGET};

  legate::mapping::detail::ProxyStoreMapping{legate::proxy::inputs, policy}.apply_inline(
    *task.impl_(),
    options,
    &policy_vectors.inputs,
    &policy_vectors.outputs,
    &policy_vectors.reductions);
  legate::mapping::detail::ProxyStoreMapping{legate::proxy::outputs[0], policy}.apply_inline(
    *task.impl_(),
    options,
    &policy_vectors.inputs,
    &policy_vectors.outputs,
    &policy_vectors.reductions);

  const auto expected       = make_instance_policy(DEFAULT_TARGET, policy);
  const auto default_policy = InstanceMappingPolicy{};

  ASSERT_THAT(policy_vectors.inputs, ::testing::ElementsAre(expected, expected));
  ASSERT_THAT(policy_vectors.outputs, ::testing::ElementsAre(expected, default_policy));
  ASSERT_THAT(policy_vectors.reductions, ::testing::ElementsAre(default_policy));
}

TEST_F(ProxyStoreMappingUnit, ApplyInlineDefaultOutputMapping)
{
  auto task           = create_task_with_all_inline_arguments();
  auto policy_vectors = make_inline_policy_vectors(task);
  const auto policy   = legate::mapping::ProxyInstanceMappingPolicy{};
  const auto options  = std::array{DEFAULT_TARGET};

  legate::mapping::detail::ProxyStoreMapping{legate::proxy::outputs, policy}.apply_inline(
    *task.impl_(),
    options,
    &policy_vectors.inputs,
    &policy_vectors.outputs,
    &policy_vectors.reductions);

  const auto expected       = make_instance_policy(DEFAULT_TARGET, policy);
  const auto default_policy = InstanceMappingPolicy{};

  ASSERT_THAT(policy_vectors.inputs, ::testing::ElementsAre(default_policy, default_policy));
  ASSERT_THAT(policy_vectors.outputs, ::testing::ElementsAre(expected, expected));
  ASSERT_THAT(policy_vectors.reductions, ::testing::ElementsAre(default_policy));
}

TEST_F(ProxyStoreMappingUnit, ApplyInlineIndexedInputMapping)
{
  auto task           = create_task_with_all_inline_arguments();
  auto policy_vectors = make_inline_policy_vectors(task);
  const auto policy   = make_proxy_policy();
  const auto options  = std::array{legate::mapping::StoreTarget::FBMEM, DEFAULT_TARGET};

  legate::mapping::detail::ProxyStoreMapping{legate::proxy::inputs[0], policy}.apply_inline(
    *task.impl_(),
    options,
    &policy_vectors.inputs,
    &policy_vectors.outputs,
    &policy_vectors.reductions);

  const auto expected       = make_instance_policy(DEFAULT_TARGET, policy);
  const auto default_policy = InstanceMappingPolicy{};

  ASSERT_THAT(policy_vectors.inputs, ::testing::ElementsAre(expected, default_policy));
  ASSERT_THAT(policy_vectors.outputs, ::testing::ElementsAre(default_policy, default_policy));
  ASSERT_THAT(policy_vectors.reductions, ::testing::ElementsAre(default_policy));
}

TEST_F(ProxyStoreMappingUnit, ApplyInlineAllReductionMapping)
{
  auto task           = create_task_with_all_inline_arguments();
  auto policy_vectors = make_inline_policy_vectors(task);
  const auto policy   = make_proxy_policy();
  const auto options  = std::array{legate::mapping::StoreTarget::FBMEM, DEFAULT_TARGET};

  legate::mapping::detail::ProxyStoreMapping{legate::proxy::reductions, policy}.apply_inline(
    *task.impl_(),
    options,
    &policy_vectors.inputs,
    &policy_vectors.outputs,
    &policy_vectors.reductions);

  const auto expected       = make_instance_policy(DEFAULT_TARGET, policy);
  const auto default_policy = InstanceMappingPolicy{};

  ASSERT_THAT(policy_vectors.inputs, ::testing::ElementsAre(default_policy, default_policy));
  ASSERT_THAT(policy_vectors.outputs, ::testing::ElementsAre(default_policy, default_policy));
  ASSERT_THAT(policy_vectors.reductions, ::testing::ElementsAre(expected));
}

TEST_F(ProxyStoreMappingUnit, ApplyInlineIndexedReductionMapping)
{
  auto task           = create_task_with_all_inline_arguments();
  auto policy_vectors = make_inline_policy_vectors(task);
  const auto policy   = make_proxy_policy();
  const auto options  = std::array{legate::mapping::StoreTarget::FBMEM, DEFAULT_TARGET};

  legate::mapping::detail::ProxyStoreMapping{legate::proxy::reductions[0], policy}.apply_inline(
    *task.impl_(),
    options,
    &policy_vectors.inputs,
    &policy_vectors.outputs,
    &policy_vectors.reductions);

  const auto expected       = make_instance_policy(DEFAULT_TARGET, policy);
  const auto default_policy = InstanceMappingPolicy{};

  ASSERT_THAT(policy_vectors.inputs, ::testing::ElementsAre(default_policy, default_policy));
  ASSERT_THAT(policy_vectors.outputs, ::testing::ElementsAre(default_policy, default_policy));
  ASSERT_THAT(policy_vectors.reductions, ::testing::ElementsAre(expected));
}

}  // namespace proxy_store_mapping_test
