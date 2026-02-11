/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/proxy_store_mapping.h>

#include <legate.h>

#include <legate/mapping/detail/proxy_store_mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/partitioning/proxy.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace proxy_store_mapping_test {

class CheckTaskTargetTask : public legate::LegateTask<CheckTaskTargetTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
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
      const auto target = ctx.input(i).data().target();

      ASSERT_EQ(target, legate::mapping::StoreTarget::SYSMEM);
    }

    {
      ASSERT_EQ(ctx.num_outputs(), 1);

      const auto target = ctx.output(0).data().target();

      ASSERT_EQ(target, legate::mapping::StoreTarget::SYSMEM);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_proxy_store_mapping";

  static void registration_callback(legate::Library library)
  {
    CheckTaskTargetTask::register_variants(library);
  }
};

class ProxyStoreMappingUnit : public RegisterOnceFixture<Config> {};

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

TEST_F(ProxyStoreMappingUnit, Apply)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, CheckTaskTargetTask::TASK_CONFIG.task_id());

  for (std::size_t i = 0; i < 2; ++i) {
    auto array = runtime->create_array(legate::Shape{1, i}, legate::int32());

    runtime->issue_fill(array, legate::Scalar{1});
    task.add_input(std::move(array));
  }

  {
    auto array = runtime->create_array(legate::Shape{2, 2}, legate::int32());

    runtime->issue_fill(array, legate::Scalar{1});
    task.add_output(std::move(array));
  }

  runtime->submit(std::move(task));
}

}  // namespace proxy_store_mapping_test
