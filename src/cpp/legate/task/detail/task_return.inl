/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_return.h>

#include <cstring>

namespace legate::detail {

inline std::size_t TaskReturn::buffer_size() const { return layout_.total_size(); }

}  // namespace legate::detail
