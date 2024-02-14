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

#include "core/mapping/mapping.h"

#include "legate.h"
#include "scoping_cffi.h"

namespace scoping {

static const char* const library_name = "scoping";

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  static constexpr int TASK_ID = ID;
};

namespace {

void validate(legate::TaskContext context)
{
  if (context.is_single_task()) {
    return;
  }

  std::int32_t num_tasks = context.get_launch_domain().get_volume();
  auto to_compare        = context.scalars().at(0).value<std::int32_t>();
  if (to_compare != num_tasks) {
    LEGATE_ABORT("Test failed: expected " << to_compare << "tasks, but got " << num_tasks
                                          << " tasks");
  }
}

void map_check(legate::TaskContext& context)
{
  std::int32_t task_count          = context.get_launch_domain().get_volume();
  std::int32_t shard_id            = legate::Processor::get_executing_processor().address_space();
  std::int32_t task_id             = context.get_task_index()[0];
  std::int32_t per_node_count      = context.scalar(0).value<std::int32_t>();
  std::int32_t proc_count          = context.scalar(1).value<std::int32_t>();
  std::int32_t start_proc_id       = context.scalar(2).value<std::int32_t>();
  std::int32_t global_proc_id      = task_id * proc_count / task_count + start_proc_id;
  std::int32_t calculated_shard_id = global_proc_id / per_node_count;
  if (shard_id != calculated_shard_id) {
    LEGATE_ABORT("Test failed: expected " << shard_id << " shard, but got " << calculated_shard_id
                                          << " shard");
  }
}

}  // namespace

class MultiVariantTask : public Task<MultiVariantTask, MULTI_VARIANT> {
 public:
  static void cpu_variant(legate::TaskContext context) { validate(context); }
#if LegateDefined(USE_OPENMP)
  static void omp_variant(legate::TaskContext context) { validate(context); }
#endif
#if LegateDefined(USE_CUDA)
  static void gpu_variant(legate::TaskContext context) { validate(context); }
#endif
};

class CpuVariantOnlyTask : public Task<CpuVariantOnlyTask, CPU_VARIANT_ONLY> {
 public:
  static void cpu_variant(legate::TaskContext context) { validate(context); }
};

class MapCheckTask : public Task<MapCheckTask, MAP_CHECK> {
 public:
  static void cpu_variant(legate::TaskContext context) { map_check(context); }
#if LegateDefined(USE_OPENMP)
  static void omp_variant(legate::TaskContext context) { map_check(context); }
#endif
#if LegateDefined(USE_CUDA)
  static void gpu_variant(legate::TaskContext context) { map_check(context); }
#endif
};

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  MultiVariantTask::register_variants(context);
  CpuVariantOnlyTask::register_variants(context);
  MapCheckTask::register_variants(context);
}

}  // namespace scoping

extern "C" {

void perform_registration(void) { scoping::registration_callback(); }
}
