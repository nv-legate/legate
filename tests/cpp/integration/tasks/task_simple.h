/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/mapping/mapping.h>

namespace task::simple {

inline constexpr std::string_view LIBRARY_NAME = "legate.simple";

extern Legion::Logger logger;

void register_tasks();

struct HelloTask : public legate::LegateTask<HelloTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};
  static void cpu_variant(legate::TaskContext context);
};

}  // namespace task::simple
