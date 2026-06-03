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

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  // Defined in saxpy.cu.
  static void gpu_variant(legate::TaskContext context);
#endif
};

// saxpy task
class SAXPYTask : public legate::LegateTask<SAXPYTask> {
 public:
  //  This `TaskConfig` signature below does the following:
  //    - Assigns a local task ID of 1 ((should be unique within the library).
  //    - Specifies that the task takes exactly 2 input stores and 1 output
  //      store; this calling convention will be checked at runtime when
  //      the task is launched.
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(
      legate::TaskSignature{}.inputs(2).outputs(1));

  // This is the CPU variant of the task
  // there could be also OpenMP and CUDA variants in Legate
  static void cpu_variant(legate::TaskContext context);

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  // Defined in saxpy.cu.
  static void gpu_variant(legate::TaskContext context);
#endif
};

// verification task
class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext context);
};
