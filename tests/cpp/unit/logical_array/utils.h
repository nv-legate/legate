/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

namespace logical_array_util_test {

[[nodiscard]] legate::LogicalArray create_array_with_type(const legate::Type& type,
                                                          bool bound,
                                                          bool nullable,
                                                          bool optimize_scalar = false);

[[nodiscard]] const legate::StructType& struct_type();

[[nodiscard]] const legate::ListType& list_type();

}  // namespace logical_array_util_test
