/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <utilities/utilities.h>
#include <utility>
#include <vector>

namespace mapping_operation_unit {

namespace {

struct MapperTaskCounts {
  static inline std::optional<std::size_t> num_reductions{};
  static inline std::optional<std::size_t> num_scalars{};

  static void reset()
  {
    num_reductions.reset();
    num_scalars.reset();
  }
};

class CheckTaskMetadataMapper final : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& /*options*/) override
  {
    MapperTaskCounts::num_reductions = task.num_reductions();
    MapperTaskCounts::num_scalars    = task.num_scalars();
    return {};
  }

  std::optional<std::size_t> allocation_pool_size(
    const legate::mapping::Task& /*task*/, legate::mapping::StoreTarget /*memory_kind*/) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    LEGATE_ABORT("This method should never be called");
    return legate::Scalar{};
  }
};

class CheckTaskMetadataTask : public legate::LegateTask<CheckTaskMetadataTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class MappingOperationUnit : public DefaultFixture {
 protected:
  static constexpr std::string_view LIBRARY_NAME = "test_mapping_operation";

  void SetUp() override
  {
    DefaultFixture::SetUp();

    auto* runtime = legate::Runtime::get_runtime();
    auto created  = false;
    auto library  = runtime->find_or_create_library(LIBRARY_NAME,
                                                    legate::ResourceConfig{},
                                                    std::make_unique<CheckTaskMetadataMapper>(),
                                                    {},
                                                    &created);

    if (created) {
      CheckTaskMetadataTask::register_variants(library);
    }
  }
};

}  // namespace

TEST_F(MappingOperationUnit, TaskReportsReductionAndScalarCounts)
{
  MapperTaskCounts::reset();

  auto* runtime       = legate::Runtime::get_runtime();
  const auto library  = runtime->find_library(LIBRARY_NAME);
  auto store          = runtime->create_store(legate::Shape{1}, legate::int64());
  constexpr auto fill = std::int64_t{1};
  auto task           = runtime->create_task(library, CheckTaskMetadataTask::TASK_CONFIG.task_id());

  runtime->issue_fill(store, legate::Scalar{fill});
  task.add_reduction(std::move(store), legate::ReductionOpKind::ADD);
  task.add_scalar_arg(legate::Scalar{fill});
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/*block=*/true);

  ASSERT_THAT(MapperTaskCounts::num_reductions, ::testing::Optional(std::size_t{1}));
  ASSERT_THAT(MapperTaskCounts::num_scalars, ::testing::Optional(std::size_t{1}));
}

}  // namespace mapping_operation_unit
