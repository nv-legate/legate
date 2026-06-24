/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/data_movement_requests.h>

namespace legate::detail {

inline void DataMovementRequests::reserve_copy_ins(std::size_t size) { copy_ins_.reserve(size); }

inline void DataMovementRequests::reserve_reduction_requests(std::size_t size)
{
  reductions_.reserve(size);
}

}  // namespace legate::detail
