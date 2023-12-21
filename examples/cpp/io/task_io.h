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

namespace legateio {

enum LegateIOOpCode {
  _OP_CODE_BASE      = 0,
  READ_EVEN_TILES    = 1,
  READ_FILE          = 2,
  READ_UNEVEN_TILES  = 3,
  WRITE_EVEN_TILES   = 4,
  WRITE_FILE         = 5,
  WRITE_UNEVEN_TILES = 6,
};

static const char* library_name = "legateio";
extern Legion::Logger logger;

void register_tasks();

struct ReadEvenTilesTask : public legate::LegateTask<ReadEvenTilesTask> {
  static const int32_t TASK_ID = READ_EVEN_TILES;
  static void cpu_variant(legate::TaskContext context);
};

struct ReadFileTask : public legate::LegateTask<ReadFileTask> {
  static const int32_t TASK_ID = READ_FILE;
  static void cpu_variant(legate::TaskContext context);
};

struct ReadUnevenTilesTask : public legate::LegateTask<ReadUnevenTilesTask> {
  static const int32_t TASK_ID = READ_UNEVEN_TILES;
  static void cpu_variant(legate::TaskContext context);
};

struct WriteEvenTilesTask : public legate::LegateTask<WriteEvenTilesTask> {
  static const int32_t TASK_ID = WRITE_EVEN_TILES;
  static void cpu_variant(legate::TaskContext context);
};

struct WriteFileTask : public legate::LegateTask<WriteFileTask> {
  static const int32_t TASK_ID = WRITE_FILE;
  static void cpu_variant(legate::TaskContext context);
};

struct WriteUnevenTilesTask : public legate::LegateTask<WriteUnevenTilesTask> {
  static const int32_t TASK_ID = WRITE_UNEVEN_TILES;
  static void cpu_variant(legate::TaskContext context);
};

}  // namespace legateio

}  // namespace task
