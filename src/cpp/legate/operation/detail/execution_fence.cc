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

#include "legate/operation/detail/execution_fence.h"

#include "legate/runtime/detail/runtime.h"

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
