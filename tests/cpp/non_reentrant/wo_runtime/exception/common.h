/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/argument_parsing/legate_args.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/zip.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utilities/env.h>
#include <utilities/utilities.h>
#include <vector>

namespace traced_exception_test {

class TracedExceptionFixture : public DefaultFixture {
  // Must disable color output, otherwise the tests checking the tracebacks will be
  // interspersed with color codes.
  //
  // Must do this in a fixture because the "should exceptions use color" check is done
  // exactly once, and on construction of a TracedException object. So we need to ensure that
  // all tests are disabling color output.
  legate::test::Environment::TemporaryEnvVar force_color_{"FORCE_COLOR", nullptr};
  legate::test::Environment::TemporaryEnvVar no_color_{"NO_COLOR", "1", true};
};

[[nodiscard]] inline std::vector<std::string> split_string(std::string_view str)
{
  std::vector<std::string> strings;
  std::string_view::size_type pos  = 0;
  std::string_view::size_type prev = 0;

  strings.reserve(static_cast<std::size_t>(std::count(str.begin(), str.end(), '\n')));
  while ((pos = str.find('\n', prev)) != std::string_view::npos) {
    strings.emplace_back(str.substr(prev, pos - prev));
    prev = pos + 1;
  }

  // To get the last substring (or only, if delimiter is not found)
  if (const auto end = str.substr(prev); !end.empty()) {
    strings.emplace_back(end);
  }
  return strings;
}

class NonStdException {
 public:
  NonStdException() = default;

  explicit NonStdException(std::string text) : text_{std::move(text)} {}

 private:
  std::string text_{};
};

MATCHER_P3(MatchesStackTrace,  // NOLINT
           exn_type_infos,     // NOLINT
           exn_messages,       // NOLINT
           file_names,         // NOLINT
           fmt::format("matches the legate stack trace (with message(s) [{}])",
                       fmt::join(exn_messages, ", ")))
{
  using ::testing::HasSubstr;
  using ::testing::MatchesRegex;
  using ::testing::StartsWith;

  const auto lines = split_string(arg);
  const auto deref = [&](std::vector<std::string>::const_iterator it) -> std::string_view {
    EXPECT_NE(it, lines.end()) << "Line-iterator out of range";
    return *it;
  };

  EXPECT_EQ(exn_type_infos.size(), exn_messages.size());
  EXPECT_EQ(exn_type_infos.size(), file_names.size());
  EXPECT_FALSE(lines.empty());
  // If cpptrace encounters an error when trying to gather a stack trace (such as failing to
  // read debug symbols from system libraries), it prints a bunch of error messages:
  //
  // Cpptrace internal error: Unable to read object file /usr/lib/libc++abi.dylib
  // Cpptrace internal error: Unable to read object file /usr/lib/system/libunwind.dylib
  // ...
  //
  // We can safely skip those to get to the actual traceback.
  auto it = std::find_if(lines.cbegin(), lines.cend(), [](std::string_view line) {
    return line.find("Cpptrace internal error") == std::string_view::npos;
  });

  EXPECT_NE(it, lines.end()) << "did not find a legate stack trace";
  EXPECT_THAT(deref(it++), MatchesRegex("LEGATE ERROR: [=]+"));
  EXPECT_THAT(deref(it++), MatchesRegex("LEGATE ERROR: System: .*"));
  EXPECT_THAT(deref(it++),
              MatchesRegex(R"(LEGATE ERROR: Legate version: [0-9]+.[0-9]+.[0-9]+ \([A-z0-9]+\))"));

  EXPECT_THAT(deref(it++),
              MatchesRegex("LEGATE ERROR: Legion version: " LEGION_VERSION R"( \([A-z0-9]+\))"));
  EXPECT_THAT(deref(it++), MatchesRegex("LEGATE ERROR: Configure options: .*"));
  EXPECT_EQ(
    deref(it++),
    fmt::format("LEGATE ERROR: LEGATE_CONFIG: {}", legate::detail::get_parsed_LEGATE_CONFIG()));
  EXPECT_THAT(deref(it++), MatchesRegex("LEGATE ERROR: [-]+"));

  EXPECT_THAT(
    deref(it++),
    MatchesRegex(
      R"(LEGATE ERROR: Exception stack contains [0-9]+ exception\(s\) \(bottom\-most exception first\):)"));
  EXPECT_EQ(deref(it++), "LEGATE ERROR:");

  for (auto&& [idx, rest] : legate::detail::enumerate(
         legate::detail::zip_equal(exn_type_infos, exn_messages, file_names))) {
    auto&& [ty_info, exn_mess, file_name] = rest;

    if (idx) {
      EXPECT_EQ(deref(it++),
                "LEGATE ERROR: The above exception was caught and rethrown with the following "
                "additional information");
    }
    EXPECT_EQ(
      deref(it++),
      fmt::format(
        "LEGATE ERROR: #{} {}: {}", idx, static_cast<const std::type_info&>(ty_info), exn_mess));
    EXPECT_EQ(deref(it++), "LEGATE ERROR:");
    EXPECT_EQ(deref(it++), "LEGATE ERROR: Stack trace (most recent call first):");

    auto found = false;

    for (; it != lines.end(); ++it) {
      auto&& val = deref(it);

      EXPECT_THAT(val, StartsWith("LEGATE ERROR:"));
      if (val.find(file_name) != std::string::npos) {
        found = true;
      }
      if (val == "LEGATE ERROR:") {
        ++it;
        break;
      }
    }

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      // Can only reliably expect to find the file name in the stack trace in debug mode. Release
      // builds will likely have stripped this out.
      EXPECT_TRUE(found);
    }
  }

  EXPECT_THAT(lines.back(), MatchesRegex("LEGATE ERROR: [=]+"));
  return true;
}

}  // namespace traced_exception_test
