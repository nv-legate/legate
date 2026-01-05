/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/proxy.h>
#include <legate/task/task.h>
#include <legate/task/task_context.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/span.h>

namespace legate::detail {

// OffloadTo is an empty task that runs with Read/Write permissions on its data,
// so that the data ends up getting copied to the target memory. Additionally, we
// modify the core-mapper to map the data to the specified target memory in
// `Runtime::offload_to()` for this task. This invalidates any other copies of the
// data in other memories and thus, frees up space.
class LEGATE_EXPORT OffloadTo : public LegateTask<OffloadTo> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::OFFLOAD_TO}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).scalars(1).redops(0).constraints(
        {Span<const legate::ProxyConstraint>{}})  // some compilers complain with {{}}
    );

  // Task body left empty because there is no computation to do. This task
  // triggers a data movement because of its R/W privileges
  static void cpu_variant(legate::TaskContext) {}

  static void gpu_variant(legate::TaskContext) {}

  static void omp_variant(legate::TaskContext) {}
};

}  // namespace legate::detail
