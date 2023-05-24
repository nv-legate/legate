/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "core/mapping/mapping.h"
#include "legate.h"

namespace provenance {

static const char* library_name = "provenance";
static legate::Logger logger(library_name);

enum TaskIDs {
  PROVENANCE = 0,
};

struct ProvenanceTask : public legate::LegateTask<ProvenanceTask> {
  static const int32_t TASK_ID = PROVENANCE;
  static void cpu_variant(legate::TaskContext& context);
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  ProvenanceTask::register_variants(context);
}

/*static*/ void ProvenanceTask::cpu_variant(legate::TaskContext& context)
{
  std::string scalar = context.scalars()[0].value<std::string>();
  auto provenance    = context.get_provenance();
  EXPECT_TRUE(provenance.find(scalar) != std::string::npos);
}

void test_manual_provenance(legate::LibraryContext* context)
{
  auto runtime           = legate::Runtime::get_runtime();
  std::string provenance = "test_manual_provenance";
  runtime->provenance_manager()->set_provenance(provenance);
  // auto task
  auto task = runtime->create_task(context, PROVENANCE);
  task->add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_push_provenance(legate::LibraryContext* context)
{
  auto runtime           = legate::Runtime::get_runtime();
  std::string provenance = "test_push_provenance";
  runtime->provenance_manager()->push_provenance(provenance);
  EXPECT_EQ(runtime->provenance_manager()->get_provenance(), provenance);
  // auto task
  auto task = runtime->create_task(context, PROVENANCE);
  task->add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_pop_provenance(legate::LibraryContext* context)
{
  auto runtime = legate::Runtime::get_runtime();
  runtime->provenance_manager()->clear_all();
  runtime->provenance_manager()->push_provenance("some provenance for provenance task");
  runtime->provenance_manager()->pop_provenance();
  // auto task
  auto task              = runtime->create_task(context, PROVENANCE);
  std::string provenance = "";
  task->add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_underflow(legate::LibraryContext* context)
{
  auto runtime = legate::Runtime::get_runtime();
  runtime->provenance_manager()->clear_all();
  runtime->provenance_manager()->set_provenance("some provenance for provenance task");
  EXPECT_THROW(runtime->provenance_manager()->pop_provenance(), std::runtime_error);
}

void test_clear_provenance(legate::LibraryContext* context)
{
  auto runtime = legate::Runtime::get_runtime();
  runtime->provenance_manager()->push_provenance("provenance for provenance task");
  runtime->provenance_manager()->push_provenance("another provenance");
  runtime->provenance_manager()->clear_all();
  // auto task
  auto task              = runtime->create_task(context, PROVENANCE);
  std::string provenance = "";
  task->add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_provenance_tracker(legate::LibraryContext* context)
{
  legate::ProvenanceTracker track(std::string(__FILE__) + ":" + std::to_string(__LINE__));
  auto runtime = legate::Runtime::get_runtime();
  // auto task
  auto task              = runtime->create_task(context, PROVENANCE);
  std::string provenance = "provenance.cc:109";
  task->add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_nested_provenance_tracker(legate::LibraryContext* context)
{
  legate::ProvenanceTracker track(std::string(__FILE__) + ":" + std::to_string(__LINE__));
  test_provenance_tracker(context);
  // The provenance string used by test_provenance_tracker should be popped out at this point
  auto runtime           = legate::Runtime::get_runtime();
  auto task              = runtime->create_task(context, PROVENANCE);
  std::string provenance = "provenance.cc:120";
  task->add_scalar_arg(legate::Scalar(provenance));
  runtime->submit(std::move(task));
}

void test_manual_tracker(legate::LibraryContext* context)
{
  legate::ProvenanceTracker track("manual provenance through tracker");
  auto runtime = legate::Runtime::get_runtime();
  // auto task
  auto task = runtime->create_task(context, PROVENANCE);
  task->add_scalar_arg(legate::Scalar(track.get_current_provenance()));
  runtime->submit(std::move(task));
}

TEST(Integration, Provenance)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  test_manual_provenance(context);
  test_push_provenance(context);
  test_pop_provenance(context);
  test_underflow(context);
  test_clear_provenance(context);
  test_provenance_tracker(context);
  test_nested_provenance_tracker(context);
  test_manual_tracker(context);
}

}  // namespace provenance
