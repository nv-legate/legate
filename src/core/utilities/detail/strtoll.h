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

#include <cerrno>
#include <cstdlib>
#include <system_error>

namespace legate::detail {

template <typename T = long long>  // NOLINT(google-runtime-int) default to match strtoll()
[[nodiscard]] T safe_strtoll(const char* env_value, char** end_ptr = nullptr)
{
  constexpr auto radix = 10;

  // must reset errno before calling std::strtoll()
  errno    = 0;
  auto ret = std::strtoll(env_value, end_ptr, radix);
  if (const auto eval = errno) {
    throw std::system_error(eval, std::generic_category(), "error occurred calling std::strtol()");
  }
  return static_cast<T>(ret);
}

}  // namespace legate::detail
