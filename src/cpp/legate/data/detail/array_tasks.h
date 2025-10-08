/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/proxy.h>
#include <legate/task/task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>

namespace legate::detail {

class Library;

class FixupRanges : public LegateTask<FixupRanges> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::FIXUP_RANGES}}.with_signature(
      legate::TaskSignature{}
        .inputs(0)
        .outputs(0, legate::TaskSignature::UNBOUNDED)
        .scalars(0)
        .redops(0)
        .constraints({Span<const legate::ProxyConstraint>{}})  // some compilers complain with {{}}
    );
  static constexpr VariantOptions GPU_VARIANT_OPTIONS =
    VariantOptions{}.with_elide_device_ctx_sync(true);

  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

void register_array_tasks(Library& core_lib);

}  // namespace legate::detail
