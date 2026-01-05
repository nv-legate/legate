/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace machine_scope {

namespace {

// NOLINTBEGIN(readability-magic-numbers)

enum TaskIDs : std::uint8_t { MULTI_VARIANT, CPU_VARIANT };

void validate(legate::TaskContext context)
{
  if (context.is_single_task()) {
    return;
  }

  const auto num_tasks  = context.get_launch_domain().get_volume();
  const auto to_compare = context.scalars().at(0).value<std::int32_t>();
  EXPECT_EQ(to_compare, num_tasks);
}

struct MultiVariantTask : public legate::LegateTask<MultiVariantTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{MULTI_VARIANT}};

  static void cpu_variant(legate::TaskContext context) { validate(context); }
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context) { validate(context); }
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context) { validate(context); }
#endif
};

struct CpuVariantOnlyTask : public legate::LegateTask<CpuVariantOnlyTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CPU_VARIANT}};

  static void cpu_variant(legate::TaskContext context) { validate(context); }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "machine_scope";

  static void registration_callback(legate::Library library)
  {
    MultiVariantTask::register_variants(library);
    CpuVariantOnlyTask::register_variants(library);
  }
};

class MachineScope : public RegisterOnceFixture<Config> {};

void test_scoping(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(library, MultiVariantTask::TASK_CONFIG.task_id());
  auto part    = task.declare_partition();

  task.add_output(store, part);
  task.add_scalar_arg(legate::Scalar{machine.count()});
  runtime->submit(std::move(task));
  if (machine.count(legate::mapping::TaskTarget::CPU) > 0) {
    const legate::Scope scope{machine.only(legate::mapping::TaskTarget::CPU)};
    auto task_scoped = runtime->create_task(library, MultiVariantTask::TASK_CONFIG.task_id());
    auto part_scoped = task_scoped.declare_partition();

    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(legate::Scalar{machine.count(legate::mapping::TaskTarget::CPU)});
    runtime->submit(std::move(task_scoped));
  }

  if (machine.count(legate::mapping::TaskTarget::OMP) > 0) {
    const legate::Scope tracker{machine.only(legate::mapping::TaskTarget::OMP)};
    auto task_scoped = runtime->create_task(library, MultiVariantTask::TASK_CONFIG.task_id());
    auto part_scoped = task_scoped.declare_partition();

    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(legate::Scalar{machine.count(legate::mapping::TaskTarget::OMP)});
    runtime->submit(std::move(task_scoped));
  }

  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    const legate::Scope tracker{machine.only(legate::mapping::TaskTarget::GPU)};
    auto task_scoped = runtime->create_task(library, MultiVariantTask::TASK_CONFIG.task_id());
    auto part_scoped = task_scoped.declare_partition();

    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(legate::Scalar{machine.count(legate::mapping::TaskTarget::GPU)});
    runtime->submit(std::move(task_scoped));
  }
}

void test_cpu_only(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(library, CpuVariantOnlyTask::TASK_CONFIG.task_id());
  auto part    = task.declare_partition();
  task.add_output(store, part);
  task.add_scalar_arg(legate::Scalar{machine.count(legate::mapping::TaskTarget::CPU)});
  runtime->submit(std::move(task));

  if (machine.count(legate::mapping::TaskTarget::CPU) > 0) {
    const legate::Scope scope{machine.only(legate::mapping::TaskTarget::CPU)};
    auto task_scoped = runtime->create_task(library, CpuVariantOnlyTask::TASK_CONFIG.task_id());
    auto part_scoped = task_scoped.declare_partition();

    task_scoped.add_output(store, part_scoped);
    task_scoped.add_scalar_arg(legate::Scalar{machine.count(legate::mapping::TaskTarget::CPU)});
    runtime->submit(std::move(task_scoped));
  }

  // checking an empty machine
  {
    EXPECT_THROW(legate::Scope{machine.only(legate::mapping::TaskTarget::CPU).slice(15, 19)},
                 std::runtime_error);
  }

  // check `slice_machine_for_task` logic
  if (machine.count(legate::mapping::TaskTarget::GPU) > 0) {
    const legate::Scope tracker{machine.only(legate::mapping::TaskTarget::GPU)};

    EXPECT_THROW(
      static_cast<void>(runtime->create_task(library, CpuVariantOnlyTask::TASK_CONFIG.task_id())),
      std::invalid_argument);
  }
}

}  // namespace

TEST_F(MachineScope, All)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  test_scoping(library);
  test_cpu_only(library);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace machine_scope
