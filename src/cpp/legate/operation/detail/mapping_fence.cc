/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/mapping_fence.h>

#include <legion.h>

namespace legate::detail {

void MappingFence::launch()
{
  Legion::Runtime::get_runtime()->issue_mapping_fence(Legion::Runtime::get_context());
}

}  // namespace legate::detail
