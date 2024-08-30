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

#include "core/operation/detail/mapping_fence.h"

#include "core/runtime/detail/runtime.h"

namespace legate::detail {

void MappingFence::launch()
{
  auto* runtime = Runtime::get_runtime();
  runtime->get_legion_runtime()->issue_mapping_fence(runtime->get_legion_context());
}

}  // namespace legate::detail
