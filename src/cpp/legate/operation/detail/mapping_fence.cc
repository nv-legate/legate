/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/mapping_fence.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void MappingFence::launch()
{
  auto* runtime = Runtime::get_runtime();
  runtime->get_legion_runtime()->issue_mapping_fence(runtime->get_legion_context());
}

}  // namespace legate::detail
