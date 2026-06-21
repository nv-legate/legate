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

// verification task
class CopyTask : public legate::LegateTask<CopyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1));

  static void cpu_variant(legate::TaskContext context);
};

// verification task
class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext context);
};
