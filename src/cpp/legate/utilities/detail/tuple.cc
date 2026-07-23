/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/tuple.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <cstdint>
#include <stdexcept>

namespace legate::detail {

void throw_invalid_tuple_sizes(std::size_t lhs_size, std::size_t rhs_size)
{
  throw TracedException<std::invalid_argument>{
    fmt::format("Operands should have the same size: {} != {}", lhs_size, rhs_size)};
}

}  // namespace legate::detail
