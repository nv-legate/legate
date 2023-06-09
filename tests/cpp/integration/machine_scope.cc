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

#include "legate.h"

namespace machine_scope {

static const char* library_name = "machine_scope";
static legate::Logger logger(library_name);

enum TaskIDs {
  MULTI_VARIANT = 0,
  CPU_VARIANT   = 1,
};

void validate(legate::TaskContext& context)
{
  if (context.is_single_task()) return;

  int32_t num_tasks = context.get_launch_domain().get_volume();
  auto to_compare   = context.scalars().at(0).value<int32_t>();
  EXPECT_EQ(to_compare, num_tasks);
}

struct MultiVariantTask : public legate::LegateTask<MultiVariantTask> {
  static const int32_t TASK_ID = MULTI_VARIANT;

  static void cpu_variant(legate::TaskContext& context) { validate(context); }
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext& context) { validate(context); }
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext& context) { validate(context); }
#endif
};

struct CpuVariantOnlyTask : public legate::LegateTask<CpuVariantOnlyTask> {
  static const int32_t TASK_ID = CPU_VARIANT;
  static void cpu_variant(legate::TaskContext& context) { validate(context); }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  MultiVariantTask::register_variants(context);
  CpuVariantOnlyTask::register_variants(context);
}

void test_scoping(legate::LibraryContext* context)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({5, 5}, legate::int64());
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(context, MULTI_VARIANT);
  auto part    = task->declare_partition();
  task->add_output(store, part);
  task->add_scalar_arg(machine.count());
  runtime->submit(std::move(task));

  if (machine.count(legate::mapping::TaskTarget::CPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::CPU));
    auto task_scoped = runtime->create_task(context, MULTI_VARIANT);
    auto part_scoped = task_scoped->declare_partition();
    task_scoped->add_output(store, part_scoped);
    task_scoped->add_scalar_arg(machine.count(legate::mapping::TaskTarget::CPU));
    runtime->submit(std::move(task_scoped));
  }

  if (machine.count(legate::mapping::TaskTarget::OMP) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::OMP));
    auto task_scoped = runtime->create_task(context, MULTI_VARIANT);
    auto part_scoped = task_scoped->declare_partition();
    task_scoped->add_output(store, part_scoped);
    task_scoped->add_scalar_arg(machine.count(legate::mapping::TaskTarget::OMP));
    runtime->submit(std::move(task_scoped));
  }

  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::GPU));
    auto task_scoped = runtime->create_task(context, MULTI_VARIANT);
    auto part_scoped = task_scoped->declare_partition();
    task_scoped->add_output(store, part_scoped);
    task_scoped->add_scalar_arg(machine.count(legate::mapping::TaskTarget::GPU));
    runtime->submit(std::move(task_scoped));
  }
}

void test_cpu_only(legate::LibraryContext* context)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({5, 5}, legate::int64());
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(context, CPU_VARIANT);
  auto part    = task->declare_partition();
  task->add_output(store, part);
  task->add_scalar_arg(machine.count(legate::mapping::TaskTarget::CPU));
  runtime->submit(std::move(task));

  if (machine.count(legate::mapping::TaskTarget::CPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::CPU));
    auto task_scoped = runtime->create_task(context, CPU_VARIANT);
    auto part_scoped = task_scoped->declare_partition();
    task_scoped->add_output(store, part_scoped);
    task_scoped->add_scalar_arg(machine.count(legate::mapping::TaskTarget::CPU));
    runtime->submit(std::move(task_scoped));
  }

  // checking an empty machine
  {
    EXPECT_THROW(
      legate::MachineTracker(machine.only(legate::mapping::TaskTarget::CPU).slice(15, 19)),
      std::runtime_error);
  }

  // check `slice_machine_for_task` logic
  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::GPU));
    EXPECT_THROW(runtime->create_task(context, CPU_VARIANT), std::invalid_argument);
  }
}

TEST(Integration, MachineScope)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  test_scoping(context);
  test_cpu_only(context);
}

}  // namespace machine_scope
