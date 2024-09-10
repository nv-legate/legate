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

#pragma once

#include <legate.h>

namespace hello_world {

class HelloWorld : public legate::LegateTask<HelloWorld> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{5};

  static void cpu_variant(legate::TaskContext);
};

}  // namespace hello_world
