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

#include "core/task/task.h"

namespace legate::detail {

class Library;

struct FixupRanges : public LegateTask<FixupRanges> {
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

struct OffsetsToRanges : public LegateTask<OffsetsToRanges> {
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

struct RangesToOffsets : public LegateTask<RangesToOffsets> {
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

void register_array_tasks(Library* core_lib);

}  // namespace legate::detail
