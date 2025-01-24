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

#include <legate/utilities/detail/error.h>

#include <legate_defines.h>

#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/span.h>
#include <legate/version.h>

#include <fmt/color.h>
#include <fmt/format.h>

#include <cpptrace/basic.hpp>
#include <cpptrace/utils.hpp>

#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <sys/utsname.h>
#include <vector>

namespace legate::detail {

ErrorDescription::ErrorDescription(std::string single_line, cpptrace::stacktrace stack_trace)
  : ErrorDescription{std::vector{std::move(single_line)}, std::move(stack_trace)}
{
}

ErrorDescription::ErrorDescription(std::vector<std::string> lines, cpptrace::stacktrace stack_trace)
  : message_lines{std::move(lines)}, trace{std::move(stack_trace)}
{
}

// ==========================================================================================

namespace {

constexpr auto BASE_ERROR_PREFIX = std::string_view{"LEGATE ERROR:"};

[[nodiscard]] bool detect_use_color() noexcept
{
  try {
    // Don't use EnvironmentVariable, because it may itself throw a TracedException. If that
    // happens, we get:
    //
    // libc++abi: __cxa_guard_acquire detected recursive initialization: do you have a
    // function-local static variable whose initialization depends on that function?
    //
    // Which is not great
    if (const auto* no_color = std::getenv("NO_COLOR"); no_color && (no_color[0] != '0')) {
      return false;
    }
    return cpptrace::isatty(cpptrace::stdout_fileno) && cpptrace::isatty(cpptrace::stderr_fileno);
  } catch (...) {
    return false;
  }
}

[[nodiscard]] std::string make_error_prefix(bool use_color)
{
  if (use_color) {
    return fmt::format(fmt::fg(fmt::color::indian_red), BASE_ERROR_PREFIX);
  }
  return std::string{BASE_ERROR_PREFIX};
}

[[nodiscard]] std::string system_summary(std::string_view prefix)
{
  struct ::utsname res = {};

  if (::uname(&res)) {
    return fmt::format("{} System: unknown system\n", prefix);
  }
  return fmt::format("{} System: {}, {}, {}, {}, {}\n",
                     prefix,
                     res.sysname,
                     res.release,
                     res.nodename,
                     res.version,
                     res.machine);
}

[[nodiscard]] std::string legate_version_summary(std::string_view prefix)
{
  return fmt::format("{} Legate version: {}.{}.{} ({})\n",
                     prefix,
                     LEGATE_VERSION_MAJOR,
                     LEGATE_VERSION_MINOR,
                     LEGATE_VERSION_PATCH,
                     LEGATE_GIT_HASH);
}

[[nodiscard]] std::string legion_version_summary(std::string_view prefix)
{
  return fmt::format("{} Legion version: {} ({})\n", prefix, LEGION_VERSION, LEGION_GIT_HASH);
}

[[nodiscard]] std::string configure_summary(std::string_view prefix)
{
  return fmt::format("{} Configure options: {}\n", prefix, LEGATE_CONFIGURE_OPTIONS);
}

void add_traceback(std::string_view PREFIX,
                   bool USE_COLOR,
                   const cpptrace::stacktrace& trace,
                   std::string* ret)
{
  fmt::format_to(std::back_inserter(*ret), "{} Stack trace (most recent call first):\n", PREFIX);
  if (trace.empty()) {
    fmt::format_to(std::back_inserter(*ret), "{} <unknown stack trace>\n", PREFIX);
    return;
  }

  const auto num_digits = [&] {
    const auto num_frames = trace.frames.size();

    if (num_frames < 10) {  // NOLINT(readability-magic-numbers)
      return 1;
    }
    if (num_frames < 100) {  // NOLINT(readability-magic-numbers)
      return 2;
    }
    if (num_frames < 1'000) {  // NOLINT(readability-magic-numbers)
      return 3;
    }
    // If we are handling tracebacks longer than 1000 function calls deep then we are in bigger
    // trouble.
    return 4;
  }();

  for (auto&& [idx, frame] : enumerate(trace)) {
    fmt::format_to(std::back_inserter(*ret),
                   "{} #{:<{}} {}\n",
                   PREFIX,
                   idx,
                   num_digits,
                   frame.to_string(USE_COLOR));
  }
}

void add_error_summary(std::string_view PREFIX,
                       bool USE_COLOR,
                       Span<const ErrorDescription> errs,
                       std::string* ret)
{
  static const auto EMPTY_LINE        = fmt::format("{}\n", PREFIX);
  static const auto NESTED_DISCLAIMER = fmt::format(
    "{} The above exception was caught and rethrown with the following additional "
    "information\n",
    PREFIX);

  fmt::format_to(std::back_inserter(*ret),
                 "{} Exception stack contains {} exception(s) (bottom-most exception first):\n",
                 PREFIX,
                 errs.size());
  for (auto&& [idx, descr] : enumerate(errs)) {
    ret->append(EMPTY_LINE);
    if (idx) {
      ret->append(NESTED_DISCLAIMER);
    }
    for (auto&& line : descr.message_lines) {
      fmt::format_to(std::back_inserter(*ret), "{} #{} {}\n", PREFIX, idx, line);
    }
    add_traceback(PREFIX, USE_COLOR, descr.trace, ret);
  }
}

}  // namespace

std::string make_error_message(Span<const ErrorDescription> errs)
{
  // If any of this throws an exception, we are basically screwed
  static const auto USE_COLOR              = detect_use_color();
  static const auto PREFIX                 = make_error_prefix(USE_COLOR);
  constexpr auto RULER_LENGTH              = 80;
  static const auto RULER                  = fmt::format("{} {:=>{}}\n", PREFIX, "", RULER_LENGTH);
  static const auto SYSTEM_SUMMARY         = system_summary(PREFIX);
  static const auto LEGATE_VERSION_SUMMARY = legate_version_summary(PREFIX);
  static const auto LEGION_VERSION_SUMMARY = legion_version_summary(PREFIX);
  static const auto CONFIGURE_SUMMARY      = configure_summary(PREFIX);
  constexpr auto TEN_KB                    = 10 * 1024;

  std::string ret;

  // With the system info, version info, color, and a moderately long stack trace, we easily
  // clear 5-6Kb in a single string here, so best to reserve this beast upfront. Note: the goal
  // is not performance, but to avoid fragmentation and subsequently running out of memory (if
  // that's even possible) due to constantly realloc-ing the buffer.
  ret.reserve(TEN_KB);
  ret += RULER;
  ret += SYSTEM_SUMMARY;
  ret += LEGATE_VERSION_SUMMARY;
  ret += LEGION_VERSION_SUMMARY;
  ret += CONFIGURE_SUMMARY;
  add_error_summary(PREFIX, USE_COLOR, errs, &ret);
  ret += RULER;
  return ret;
}

}  // namespace legate::detail
