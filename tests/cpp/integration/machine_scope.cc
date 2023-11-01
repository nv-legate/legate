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

#include "legate.h"
#include "utilities/utilities.h"

namespace machine_scope {

using Integration = DefaultFixture;

static const char* library_name = "machine_scope";
static legate::Logger logger(library_name);

enum TaskIDs {
  MULTI_VARIANT = 0,
  CPU_VARIANT   = 1,
};

void validate(legate::TaskContext context)
{
  if (context.is_single_task()) return;

  int32_t num_tasks = context.get_launch_domain().get_volume();
  auto to_compare   = context.scalars().at(0).value<int32_t>();
  EXPECT_EQ(to_compare, num_tasks);
}

struct MultiVariantTask : public legate::LegateTask<MultiVariantTask> {
  static const int32_t TASK_ID = MULTI_VARIANT;

  static void cpu_variant(legate::TaskContext context) { validate(context); }
#if LegateDefined(USE_OPENMP)
  static void omp_variant(legate::TaskContext context) { validate(context); }
#endif
#if LegateDefined(USE_CUDA)
  static void gpu_variant(legate::TaskContext context) { validate(context); }
#endif
};

struct CpuVariantOnlyTask : public legate::LegateTask<CpuVariantOnlyTask> {
  static const int32_t TASK_ID = CPU_VARIANT;
  static void cpu_variant(legate::TaskContext context) { validate(context); }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  MultiVariantTask::register_variants(library);
  CpuVariantOnlyTask::register_variants(library);
}

void test_scoping(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(library, MULTI_VARIANT);
  auto part    = task.declare_partition();
  task.add_output(store, part);
  task.add_scalar_arg(machine.count());
  runtime->submit(std::move(task));

  if (machine.count(legate::mapping::TaskTarget::CPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::CPU));
    auto task_scoped = runtime->create_task(library, MULTI_VARIANT);
    auto part_scoped = task_scoped.declare_partition();
    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(machine.count(legate::mapping::TaskTarget::CPU));
    runtime->submit(std::move(task_scoped));
  }

  if (machine.count(legate::mapping::TaskTarget::OMP) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::OMP));
    auto task_scoped = runtime->create_task(library, MULTI_VARIANT);
    auto part_scoped = task_scoped.declare_partition();
    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(machine.count(legate::mapping::TaskTarget::OMP));
    runtime->submit(std::move(task_scoped));
  }

  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::GPU));
    auto task_scoped = runtime->create_task(library, MULTI_VARIANT);
    auto part_scoped = task_scoped.declare_partition();
    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(machine.count(legate::mapping::TaskTarget::GPU));
    runtime->submit(std::move(task_scoped));
  }
}

void test_cpu_only(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(library, CPU_VARIANT);
  auto part    = task.declare_partition();
  task.add_output(store, part);
  task.add_scalar_arg(machine.count(legate::mapping::TaskTarget::CPU));
  runtime->submit(std::move(task));

  if (machine.count(legate::mapping::TaskTarget::CPU) > 0) {
    legate::MachineTracker tracker(machine.only(legate::mapping::TaskTarget::CPU));
    auto task_scoped = runtime->create_task(library, CPU_VARIANT);
    auto part_scoped = task_scoped.declare_partition();
    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(machine.count(legate::mapping::TaskTarget::CPU));
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
    EXPECT_THROW(static_cast<void>(runtime->create_task(library, CPU_VARIANT)),
                 std::invalid_argument);
  }
}

TEST_F(Integration, MachineScope)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  test_scoping(library);
  test_cpu_only(library);
}

}  // namespace machine_scope
