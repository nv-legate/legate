/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/buffer.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <cstddef>
#include <stdexcept>

namespace legate::detail {

void check_alignment(std::size_t alignment)
{
  if (alignment == 0) {
    throw detail::TracedException<std::domain_error>{"alignment cannot be 0"};
  }

  constexpr auto is_power_of_2 = [](std::size_t n) { return (n & (n - 1)) == 0; };

  if (!is_power_of_2(alignment)) {
    throw detail::TracedException<std::domain_error>{
      fmt::format("invalid alignment {}, must be a power of 2", alignment)};
  }
}

}  // namespace legate::detail
