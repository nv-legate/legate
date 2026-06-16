/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <cstdint>
#include <iostream>
#include <string_view>

class InitTask : public legate::LegateTask<InitTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(legate::TaskSignature{}.outputs(1));

  static void cpu_variant(legate::TaskContext context);
};

class ReductionTask : public legate::LegateTask<ReductionTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(
      legate::TaskSignature{}.inputs(1).redops(1));

  static void cpu_variant(legate::TaskContext context);

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};
