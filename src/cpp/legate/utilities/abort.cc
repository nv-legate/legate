/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/utilities/abort.h>

#include <legate/comm/detail/coll.h>
#include <legate/utilities/detail/error.h>

#include <fmt/format.h>

#include <cpptrace/basic.hpp>

#include <array>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace legate::detail {

void abort_handler(std::string_view file, std::string_view func, int line, std::stringstream* ss)
{
  const std::array<ErrorDescription, 1> errs = {ErrorDescription{
    std::vector{fmt::format("Legate called abort at {}:{} in {}()", file, line, func),
                std::move(*ss).str()},
    cpptrace::stacktrace::current(/* skip */ 1)}};

  std::cerr << make_error_message({errs.cbegin(), errs.cend()})
            << std::endl;  // NOLINT(performance-avoid-endl)
  comm::coll::abort();
}

}  // namespace legate::detail
