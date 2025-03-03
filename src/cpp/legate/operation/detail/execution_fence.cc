/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/execution_fence.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void ExecutionFence::launch()
{
  auto* runtime = Runtime::get_runtime();
  if (const auto future =
        runtime->get_legion_runtime()->issue_execution_fence(runtime->get_legion_context());
      block_) {
    future.wait();
  }
}

}  // namespace legate::detail
