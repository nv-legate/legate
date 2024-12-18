/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/comm/detail/coll.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/error.h>

#include <array>
#include <cpptrace/basic.hpp>
#include <fmt/format.h>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace legate::detail {

void abort_handler(std::string_view file, std::string_view func, int line, std::stringstream* ss)
{
  const std::array<std::string, 2> exn_messages = {
    fmt::format("Legate called abort at {}:{} in {}()", file, line, func), std::move(*ss).str()};

  std::cerr << make_error_message({exn_messages.begin(), exn_messages.end()},
                                  cpptrace::stacktrace::current(/* skip */ 1))
            << std::endl;  // NOLINT(performance-avoid-endl)
  comm::coll::abort();
}

}  // namespace legate::detail
