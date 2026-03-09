/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_array.h>

namespace legate::detail {

inline bool LogicalArray::needs_flush() const { return is_mapped(); }

}  // namespace legate::detail
