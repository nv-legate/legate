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

#include "task_region_manager.h"

namespace task {

namespace region_manager {

/*static*/ void TesterTask::cpu_variant(legate::TaskContext context)
{
  auto outputs = context.outputs();
  for (auto&& output : outputs) {
    auto store = output.data();
    if (store.is_unbound_store()) {
      store.bind_empty_data();
    }
  }
}

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  auto options = legate::VariantOptions{}.with_return_size(8192);
  TesterTask::register_variants(context, {{LEGATE_CPU_VARIANT, options}});
}

}  // namespace region_manager

}  // namespace task
