/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/logging.h>

#include <legate/comm/detail/logger.h>
#include <legate/mapping/detail/base_mapper.h>
#include <legate/runtime/detail/argument_parsing/util.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/cpp_version.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>

#include <realm/logging.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace legate::detail {

namespace {

constexpr auto LOG_LEVEL_CHOICES =
  std::array<std::tuple<std::string_view, Realm::Logger::LoggingLevel, std::string_view>, 5>{
    std::make_tuple("all",
                    Realm::Logger::LoggingLevel::LEVEL_SPEW,
                    "Enable any and all logging messages (warning: extremely verbose)"),
    std::make_tuple("debug",
                    Realm::Logger::LoggingLevel::LEVEL_DEBUG,
                    "Enable logging message of debug severity or higher"),
    std::make_tuple("info",
                    Realm::Logger::LoggingLevel::LEVEL_INFO,
                    "Enable logging messages of informational severity or higher"),
    std::make_tuple("error",
                    Realm::Logger::LoggingLevel::LEVEL_ERROR,
                    "Enable logging messages of error severity or higher"),
    std::make_tuple("none",
                    Realm::Logger::LoggingLevel::LEVEL_NONE,
                    "Print no logging messages. Note, this is not equivalent to not setting up the "
                    "logger. If this option is selected, no logging messages of any kind "
                    "(including error messages) will be printed")};

[[nodiscard]] std::pair<std::string_view, std::string_view> logger_name_and_level(
  std::string_view spec)
{
  const auto eq_pos = spec.find('=');

  if (eq_pos == std::string_view::npos) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid logger specification '{}', does not contain an '='. Expected "
                  "'logger_name=value'.",
                  spec)};
  }
  if (eq_pos == 0) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Invalid logger specification '{}', has no logger name. Expected 'logger_name=value'.",
      spec)};
  }
  return {spec.substr(0, eq_pos), spec.substr(eq_pos + 1)};
}

[[nodiscard]] bool is_number(std::string_view s)
{
  return !s.empty() && std::all_of(s.cbegin(), s.cend(), ::isdigit);
}

void maybe_numeric_log_level(std::string_view level, std::string_view spec, std::string* ret)
{
  if (is_number(level)) {
    fmt::format_to(std::back_inserter(*ret), "{},", spec);
    return;
  }

  std::string choices;

  for (auto&& [choice, _, _2] : LOG_LEVEL_CHOICES) {
    fmt::format_to(std::back_inserter(choices), "{}, ", choice);
  }
  LEGATE_ASSERT(choices.back() == ' ');
  choices.pop_back();
  LEGATE_ASSERT(choices.back() == ',');
  choices.pop_back();

  throw TracedException<std::invalid_argument>{fmt::format(
    "Unknown logging level '{}' (from '{}'), expected one of [{}]", level, spec, choices)};
}

}  // namespace

std::string convert_log_levels(std::string_view log_levels)
{
  if (log_levels.empty()) {
    return {};
  }

  constexpr auto LEVELS_BEGIN = LOG_LEVEL_CHOICES.begin();
  constexpr auto LEVELS_END   = LOG_LEVEL_CHOICES.end();

  // Would have done
  //
  // const auto [logger_name, level] = logger_name_and_level(spec);
  // const auto it                   = std::find_if(
  //    LEVELS_BEGIN, LEVELS_END, [&](const auto& pair) { return pair.first == level; });
  //
  // But apparently capturing "level" in the second lambda is a C++20 extension...
  LEGATE_CPP_VERSION_TODO(20, "Use the above instead of extra lambda");
  constexpr auto find_level = [](std::string_view level) {
    return std::find_if(
      LEVELS_BEGIN, LEVELS_END, [&](const auto& tup) { return std::get<0>(tup) == level; });
  };

  std::string ret;
  // This will over-reserve (since we are converting e.g. 'debug' to '1') but that's OK, what's
  // a little extra memory between friends?
  ret.reserve(log_levels.size());
  for (auto&& spec : string_split<std::string_view>(log_levels, ',')) {
    const auto [logger_name, level] = logger_name_and_level(spec);

    if (const auto it = find_level(level); it == LEVELS_END) {
      // Silently support the numbered logging levels as well
      maybe_numeric_log_level(level, spec, &ret);
    } else {
      fmt::format_to(std::back_inserter(ret),
                     "{}={},",
                     logger_name,
                     fmt::underlying(std::get<Realm::Logger::LoggingLevel>(*it)));
    }
  }
  // Remove the final ',' from the last loop iteration
  LEGATE_CHECK(ret.back() == ',');
  ret.pop_back();
  return ret;
}

std::string logging_help_str()
{
  std::string ret =
    "Comma separated list of loggers to enable and their "
    "level: logger_name=level. For example: legate=debug,foo=info,bar=warning. Level must be "
    "one of:\n";

  for (auto&& [choice, _, help_str] : LOG_LEVEL_CHOICES) {
    fmt::format_to(std::back_inserter(ret), "\n{} - {}.", choice, help_str);
  }

  fmt::format_to(std::back_inserter(ret),
                 "\n"
                 "\n"
                 "Available legate loggers are:\n"
                 "- {} (the core legate logger)\n"
                 "- {} (the store partitioning logger)\n"
                 "- {} (the mapping logger)\n"
                 "- {} (the collective communication logger)",
                 log_legate().get_name(),
                 log_legate_partitioner().get_name(),
                 mapping::detail::BaseMapper::LOGGER_NAME,
                 comm::coll::logger().get_name());
  return ret;
}

}  // namespace legate::detail
