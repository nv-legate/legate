/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

// These are forward declared in tuple.h to avoid including this header
// NOLINTBEGIN(readability-redundant-declaration)
[[noreturn]] LEGATE_EXPORT void throw_invalid_tuple_sizes(std::size_t lhs_size,
                                                          std::size_t rhs_size);
// NOLINTEND(readability-redundant-declaration)

}  // namespace legate::detail
