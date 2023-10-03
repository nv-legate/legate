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

#include <gtest/gtest.h>

#include "legate.h"
#include "tasks/task_region_manager.h"

namespace region_manager {

TEST(Integration, RegionManager)
{
  task::region_manager::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::region_manager::library_name);
  auto task    = runtime->create_task(context, 0);

  std::vector<legate::LogicalStore> stores;
  for (uint32_t idx = 0; idx < LEGION_MAX_FIELDS * 2; ++idx) {
    auto store = runtime->create_store({10}, legate::int64());
    auto part  = task.declare_partition();
    task.add_output(store, part);
    stores.push_back(store);
  }
  runtime->submit(std::move(task));
}

}  // namespace region_manager
