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

#pragma once

#include "legion.h"

namespace legate::detail {

class Library;

void register_legate_core_sharding_functors(Legion::Runtime* runtime,
                                            const detail::Library* core_library);

[[nodiscard]] Legion::ShardingID find_sharding_functor_by_projection_functor(
  Legion::ProjectionID proj_id);

}  // namespace legate::detail
