/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/data/buffer.h>
#include <legate/utilities/detail/traced_exception.h>

#include <cstddef>
#include <fmt/format.h>
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
