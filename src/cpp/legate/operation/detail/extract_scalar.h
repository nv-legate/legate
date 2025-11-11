/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/task/detail/legion_task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/macros.h>

#include <legion/api/types.h>

#include <vector>

namespace legate::detail {

class ExtractScalar : public LegionTask<ExtractScalar> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::EXTRACT_SCALAR}}
      .with_variant_options(
        legate::VariantOptions{}.with_has_allocations(true).with_elide_device_ctx_sync(true))
      .with_signature(legate::TaskSignature{}.inputs(0).outputs(0).scalars(2));

  [[nodiscard]] static Legion::UntypedDeferredValue cpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  [[nodiscard]] static Legion::UntypedDeferredValue omp_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);
#endif

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  [[nodiscard]] static Legion::UntypedDeferredValue gpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);
#endif
};

}  // namespace legate::detail
