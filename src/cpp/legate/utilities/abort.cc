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

#include "legate/utilities/abort.h"

#include "legate/comm/coll.h"
#include "legate/utilities/typedefs.h"

#include <legion.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>

namespace legate::detail {

void abort_handler(std::string_view file, std::string_view func, int line, std::stringstream* ss)
{
  const auto print_message = [&](auto&& dest) -> decltype(auto) {
    // Flush any output first so our message definitely shows up
    std::cout << std::flush;

    dest << "Legate called abort at " << file << ":" << line << " in " << func
         << "(): " << std::move(*ss).str();
    return std::forward<std::decay_t<decltype(dest)>>(dest);
  };

  // If Legion is not yet initialized (or it has shut down), then the Legion loggers silently
  // swallow the error messages. By using std::cerr, we lose out on some of the nice formatting
  // (e.g. printing the current node ID), but at least the abort message gets printed.
  if (Legion::Runtime::has_runtime() && Legion::Runtime::has_context()) {
    print_message(log_legate().error());
  } else {
    print_message(std::cerr) << std::endl;  // NOLINT(performance-avoid-endl)
  }
  legate::comm::coll::collAbort();
  // if we are here, then either the comm library has not been initialized, or it didn't have
  // an abort mechanism.Either way, we abort normally now.
  std::abort();
}

}  // namespace legate::detail
