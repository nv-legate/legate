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

#include <gtest/gtest.h>

#include "core/runtime/detail/runtime.h"
#include "legate.h"
#include "utilities/utilities.h"

namespace provenance {

using Integration = DefaultFixture;

static const char* library_name = "test_provenance";

enum TaskIDs {
  PROVENANCE = 0,
};

struct ProvenanceTask : public legate::LegateTask<ProvenanceTask> {
  static const int32_t TASK_ID = PROVENANCE;
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

void test_manual_provenance(legate::Library library)
{
  auto runtime           = legate::Runtime::get_runtime();
  std::string provenance = "test_manual_provenance";
  runtime->impl()->provenance_manager()->set_provenance(provenance);
  // auto task
  auto task = runtime->create_task(library, PROVENANCE);
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_push_provenance(legate::Library library)
{
  auto runtime           = legate::Runtime::get_runtime();
  std::string provenance = "test_push_provenance";
  runtime->impl()->provenance_manager()->push_provenance(provenance);
  EXPECT_EQ(runtime->impl()->provenance_manager()->get_provenance(), provenance);
  // auto task
  auto task = runtime->create_task(library, PROVENANCE);
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_pop_provenance(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  runtime->impl()->provenance_manager()->clear_all();
  runtime->impl()->provenance_manager()->push_provenance("some provenance for provenance task");
  runtime->impl()->provenance_manager()->pop_provenance();
  // auto task
  auto task              = runtime->create_task(library, PROVENANCE);
  std::string provenance = "";
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_underflow(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  runtime->impl()->provenance_manager()->clear_all();
  runtime->impl()->provenance_manager()->set_provenance("some provenance for provenance task");
  EXPECT_THROW(runtime->impl()->provenance_manager()->pop_provenance(), std::runtime_error);
}

void test_clear_provenance(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  runtime->impl()->provenance_manager()->push_provenance("provenance for provenance task");
  runtime->impl()->provenance_manager()->push_provenance("another provenance");
  runtime->impl()->provenance_manager()->clear_all();
  // auto task
  auto task              = runtime->create_task(library, PROVENANCE);
  std::string provenance = "";
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_provenance_tracker(legate::Library library)
{
  legate::ProvenanceTracker track(std::string(__FILE__) + ":" + std::to_string(__LINE__));
  auto runtime = legate::Runtime::get_runtime();
  // auto task
  auto task              = runtime->create_task(library, PROVENANCE);
  std::string provenance = "provenance.cc:107";
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_nested_provenance_tracker(legate::Library library)
{
  legate::ProvenanceTracker track(std::string(__FILE__) + ":" + std::to_string(__LINE__));
  test_provenance_tracker(library);
  // The provenance string used by test_provenance_tracker should be popped out at this point
  auto runtime           = legate::Runtime::get_runtime();
  auto task              = runtime->create_task(library, PROVENANCE);
  std::string provenance = "provenance.cc:118";
  task.add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_manual_tracker(legate::Library library)
{
  legate::ProvenanceTracker track("manual provenance through tracker");
  auto runtime = legate::Runtime::get_runtime();
  // auto task
  auto task = runtime->create_task(library, PROVENANCE);
  task.add_scalar_arg(legate::Scalar(track.get_current_provenance()));
  runtime->submit(std::move(task));
}

TEST_F(Integration, Provenance)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  test_manual_provenance(library);
  test_push_provenance(library);
  test_pop_provenance(library);
  test_underflow(library);
  test_clear_provenance(library);
  test_provenance_tracker(library);
  test_nested_provenance_tracker(library);
  test_manual_tracker(library);
}

}  // namespace provenance
