/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/runtime/detail/runtime.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace provenance {

using Integration = DefaultFixture;

static const char* library_name = "test_provenance";

enum TaskIDs {
  PROVENANCE = 0,
};

struct ProvenanceTask : public legate::LegateTask<ProvenanceTask> {
  static const std::int32_t TASK_ID = PROVENANCE;
  static void cpu_variant(legate::TaskContext context);
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  ProvenanceTask::register_variants(library);
}

/*static*/ void ProvenanceTask::cpu_variant(legate::TaskContext context)
{
  std::string scalar = context.scalar(0).value<std::string>();
  auto provenance    = context.get_provenance();
  EXPECT_TRUE(provenance.find(scalar) != std::string::npos);
}

void test_provenance(legate::Library library)
{
  const auto provenance = std::string(__FILE__) + ":" + std::to_string(__LINE__);
  legate::Scope scope{provenance};
  auto runtime = legate::Runtime::get_runtime();
  // auto task
  auto task = runtime->create_task(library, PROVENANCE);
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_nested_provenance(legate::Library library)
{
  const auto provenance = std::string(__FILE__) + ":" + std::to_string(__LINE__);
  legate::Scope scope{provenance};
  test_provenance(library);
  // The provenance string used by test_provenance should be popped out at this point
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, PROVENANCE);
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

TEST_F(Integration, Provenance)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  test_provenance(library);
  test_nested_provenance(library);
}

}  // namespace provenance
