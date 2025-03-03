/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/util.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/base.h>

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace legate::detail {

template <typename StringType>
std::vector<StringType> string_split(std::string_view command, const char sep)
{
  std::vector<StringType> qargs;

  LEGATE_ASSERT(sep != '\"' && sep != '\'');
  while (!command.empty()) {
    std::size_t arglen;
    auto quoted = false;

    if (const auto c = command.front(); c == '\"' || c == '\'') {
      command.remove_prefix(1);
      quoted = true;
      arglen = command.find(c);
      if (arglen == std::string_view::npos) {
        throw TracedException<std::invalid_argument>{
          fmt::format("Unterminated quote: '{}'", command)};
      }
    } else if (c == sep) {
      command.remove_prefix(1);
      continue;
    } else {
      arglen = std::min(command.find(sep), command.size());
    }

    if (const auto sub = command.substr(0, arglen); !sub.empty()) {
      qargs.emplace_back(sub);
    }
    command.remove_prefix(arglen + quoted);
  }
  return qargs;
}

template std::vector<std::string> string_split(std::string_view command, const char sep);
template std::vector<std::string_view> string_split(std::string_view command, const char sep);

}  // namespace legate::detail
