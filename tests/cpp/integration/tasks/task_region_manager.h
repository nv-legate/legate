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

#pragma once

#include "core/mapping/mapping.h"

#include "legate.h"

namespace task {

namespace region_manager {

enum TaskOpCode {
  PROVENANCE = 0,
};

static const char* library_name = "test_region_manager";

void register_tasks();

struct TesterTask : public legate::LegateTask<TesterTask> {
  static const std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext context);
};

}  // namespace region_manager

}  // namespace task
