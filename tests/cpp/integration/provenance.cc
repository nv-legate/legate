/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/runtime/detail/runtime.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

namespace provenance {

namespace {

// NOLINTBEGIN(readability-magic-numbers)

struct ProvenanceTask : public legate::LegateTask<ProvenanceTask> {
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext context);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_provenance";
  static void registration_callback(legate::Library library)
  {
    ProvenanceTask::register_variants(library);
  }
};

class ProvenanceTest : public RegisterOnceFixture<Config> {};

/*static*/ void ProvenanceTask::cpu_variant(legate::TaskContext context)
{
  const std::string scalar = context.scalar(0).value<std::string>();
  const auto& provenance   = context.get_provenance();

  EXPECT_TRUE(provenance.find(scalar) != std::string::npos);
}

void test_provenance_auto(legate::Library library)
{
  const auto provenance = fmt::format("{}:{}", __FILE__, __LINE__);
  const legate::Scope scope{provenance};
  auto runtime = legate::Runtime::get_runtime();
  // auto task
  auto task = runtime->create_task(library, ProvenanceTask::TASK_ID);
  task.add_scalar_arg(legate::Scalar{provenance});
  ASSERT_EQ(task.provenance(), provenance);

  runtime->submit(std::move(task));
}

void test_provenance_manual(legate::Library library)
{
  const auto provenance = fmt::format("{}:{}", __FILE__, __LINE__);
  const legate::Scope scope{provenance};
  auto runtime = legate::Runtime::get_runtime();
  // manual task
  auto task =
    runtime->create_task(library, ProvenanceTask::TASK_ID, legate::tuple<std::uint64_t>{4, 2});
  task.add_scalar_arg(legate::Scalar{provenance});
  ASSERT_EQ(task.provenance(), provenance);

  runtime->submit(std::move(task));
}

void test_nested_provenance_auto(legate::Library library)
{
  const auto provenance = fmt::format("{}:{}", __FILE__, __LINE__);
  const legate::Scope scope{provenance};
  test_provenance_auto(library);
  // The provenance string used by test_provenance should be popped out at this point
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, ProvenanceTask::TASK_ID);
  task.add_scalar_arg(legate::Scalar{provenance});
  ASSERT_EQ(task.provenance(), provenance);

  runtime->submit(std::move(task));
}

}  // namespace

TEST_F(ProvenanceTest, All)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  test_provenance_auto(library);
  test_provenance_manual(library);
  test_nested_provenance_auto(library);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace provenance
