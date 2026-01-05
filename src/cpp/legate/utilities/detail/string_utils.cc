/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/string_utils.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/base.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace legate::detail {

std::string string_to_lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

namespace {

constexpr bool is_not_space(unsigned char ch) { return !std::isspace(ch); }

}  // namespace

std::string string_lstrip(std::string s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), is_not_space));
  return s;
}

std::string string_rstrip(std::string s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), is_not_space).base(), s.end());
  return s;
}

std::string string_strip(std::string s) { return string_lstrip(string_rstrip(std::move(s))); }

std::string_view string_remove_prefix(std::string_view s, std::string_view prefix)
{
  if (const auto pos = s.find(prefix); pos == 0) {
    s.remove_prefix(prefix.size());
  }
  return s;
}

template <typename StringType>
std::vector<StringType> string_split(std::string_view sv, const char sep)
{
  std::vector<StringType> ret;

  LEGATE_CHECK(sep != '\"' && sep != '\'');
  while (!sv.empty()) {
    const auto c = sv.front();

    if (c == sep) {
      sv.remove_prefix(1);
      continue;
    }

    std::size_t arglen;
    auto quoted = false;

    if (c == '\"' || c == '\'') {
      sv.remove_prefix(1);
      quoted = true;
      arglen = sv.find(c);
      if (arglen == std::string_view::npos) {
        throw TracedException<std::invalid_argument>{fmt::format("Unterminated quote: '{}'", sv)};
      }
    } else {
      arglen = std::min(sv.find(sep), sv.size());
    }

    if (const auto sub = sv.substr(0, arglen); !sub.empty()) {
      ret.emplace_back(sub);
    }
    sv.remove_prefix(arglen + quoted);
  }
  return ret;
}

#ifndef DOXYGEN
template std::vector<std::string> string_split(std::string_view command, const char sep);
template std::vector<std::string_view> string_split(std::string_view command, const char sep);
#endif

}  // namespace legate::detail
