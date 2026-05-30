/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_single_controller_execution_slice_task {

namespace {

constexpr std::uint64_t MIN_INDEX_TASK_CPUS = 2;

class SingleControllerExecutionSliceTask : public DefaultFixture {
 protected:
  void SetUp() override
  {
    ASSERT_NO_THROW(legate::start());
    DefaultFixture::SetUp();
  }

  void TearDown() override
  {
    DefaultFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  legate::test::Environment::TemporaryEnvVar legate_config_{
    /*name=*/"LEGATE_CONFIG",
    /*value=*/"--single-controller-execution --cpus 1",
    /*overwrite=*/true};
};

}  // namespace

TEST_F(SingleControllerExecutionSliceTask, IndexTaskCompletesWithSingleControllerExecution)
{
  constexpr std::uint64_t extent = 4;
  const auto low_offsets         = std::array<std::uint64_t, 1>{1};
  const auto high_offsets        = std::array<std::uint64_t, 1>{1};
  auto* const runtime            = legate::Runtime::get_runtime();

  if (runtime->get_machine().count(legate::mapping::TaskTarget::CPU) < MIN_INDEX_TASK_CPUS) {
    GTEST_SKIP() << "Test requires at least two CPU targets";
  }

  auto store = runtime->create_store(legate::Shape{extent}, legate::int64());

  runtime->prefetch_bloated_instances(store, low_offsets, high_offsets, /*initialize=*/true);
  runtime->issue_execution_fence(/*block=*/true);
}

TEST_F(SingleControllerExecutionSliceTask, CopyUsesGlobalMachine)
{
  constexpr std::uint64_t extent = 4;
  constexpr std::int64_t value   = 123;
  auto* const runtime            = legate::Runtime::get_runtime();
  auto source                    = runtime->create_store(legate::Shape{extent}, legate::int64());
  auto target                    = runtime->create_store(legate::Shape{extent}, legate::int64());

  runtime->issue_fill(source, legate::Scalar{value});
  runtime->issue_copy(target, source);
  runtime->issue_execution_fence(/*block=*/true);

  const auto accessor        = target.get_physical_store().span_read_accessor<std::int64_t, 1>();
  const auto accessor_extent = static_cast<std::size_t>(accessor.extent(0));
  for (std::size_t idx = 0; idx < accessor_extent; ++idx) {
    ASSERT_EQ(accessor(idx), value);
  }
}

}  // namespace test_single_controller_execution_slice_task
