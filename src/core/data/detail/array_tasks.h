/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/legate_c.h"
#include "core/task/task.h"
#include "core/task/variant_options.h"

namespace legate::detail {

class Library;

class FixupRanges : public LegateTask<FixupRanges> {
 public:
  static constexpr std::int32_t TASK_ID = LEGATE_CORE_FIXUP_RANGES;
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

class OffsetsToRanges : public LegateTask<OffsetsToRanges> {
 public:
  static constexpr std::int32_t TASK_ID = LEGATE_CORE_OFFSETS_TO_RANGES;
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

class RangesToOffsets : public LegateTask<RangesToOffsets> {
 public:
  static constexpr std::int32_t TASK_ID = LEGATE_CORE_RANGES_TO_OFFSETS;
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

void register_array_tasks(Library* core_lib);

}  // namespace legate::detail
