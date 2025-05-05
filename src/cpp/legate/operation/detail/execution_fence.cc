/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/execution_fence.h>

#include <legion.h>

namespace legate::detail {

void ExecutionFence::launch()
{
  const auto future =
    Legion::Runtime::get_runtime()->issue_execution_fence(Legion::Runtime::get_context());

  if (block_) {
    future.wait();
  }
}

}  // namespace legate::detail
