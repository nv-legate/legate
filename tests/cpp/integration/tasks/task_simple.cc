/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <integration/tasks/task_simple.h>

#include <gtest/gtest.h>

namespace task::simple {

// NOLINTBEGIN(readability-magic-numbers)

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(LIBRARY_NAME);
  HelloTask::register_variants(context);
}

/*static*/ void HelloTask::cpu_variant(legate::TaskContext context)
{
  auto output = context.output(0).data();
  auto shape  = output.shape<2>();

  if (shape.empty()) {
    return;
  }

  auto acc = output.write_accessor<std::int64_t, 2>(shape);
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] + ((*it)[1] * 1000);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace task::simple
