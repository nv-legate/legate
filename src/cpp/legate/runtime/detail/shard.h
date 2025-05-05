/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/machine.h>

#include <legion.h>

namespace legate::detail {

class Library;

void register_legate_core_sharding_functors(const detail::Library& core_library);

[[nodiscard]] Legion::ShardingID find_sharding_functor_by_projection_functor(
  Legion::ProjectionID proj_id);

void create_sharding_functor_using_projection(Legion::ShardingID shard_id,
                                              Legion::ProjectionID proj_id,
                                              const mapping::ProcessorRange& range);

}  // namespace legate::detail
