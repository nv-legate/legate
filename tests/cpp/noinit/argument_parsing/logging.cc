/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/logging.h>

#include <realm/logging.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utilities/utilities.h>

namespace argument_parsing_logging_test {

class ArgumentParsingLoggingUnit
  : public DefaultFixture,
    public ::testing::WithParamInterface<
      std::tuple<std::string_view, Realm::Logger::LoggingLevel, bool>> {};

INSTANTIATE_TEST_SUITE_P(
  ,
  ArgumentParsingLoggingUnit,
  ::testing::Values(std::make_tuple("all", Realm::Logger::LoggingLevel::LEVEL_SPEW, true),
                    std::make_tuple("debug", Realm::Logger::LoggingLevel::LEVEL_DEBUG, true),
                    std::make_tuple("info", Realm::Logger::LoggingLevel::LEVEL_INFO, true),
                    std::make_tuple("print", Realm::Logger::LoggingLevel::LEVEL_PRINT, false),
                    std::make_tuple("warning", Realm::Logger::LoggingLevel::LEVEL_WARNING, false),
                    std::make_tuple("error", Realm::Logger::LoggingLevel::LEVEL_ERROR, true),
                    std::make_tuple("fatal", Realm::Logger::LoggingLevel::LEVEL_FATAL, false),
                    std::make_tuple("none", Realm::Logger::LoggingLevel::LEVEL_NONE, true)));

TEST_P(ArgumentParsingLoggingUnit, ConvertLogLevel)
{
  auto&& [str, int_val, should_handle] = GetParam();
  const auto spec                      = fmt::format("legate={}", str);

  if (should_handle) {
    const auto expected  = fmt::format("legate={}", fmt::underlying(int_val));
    const auto converted = legate::detail::convert_log_levels(spec);

    ASSERT_EQ(converted, expected);
  } else {
    ASSERT_THAT(
      [&] { return legate::detail::convert_log_levels(spec); },
      ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(fmt::format(
        "Unknown logging level '{}' (from '{}'), expected one of [all, debug, info, error, none]",
        str,
        spec))));
  }
}

TEST_P(ArgumentParsingLoggingUnit, ConvertLogLevelAllowInt)
{
  auto&& [_, int_val, _2] = GetParam();
  const auto spec         = fmt::format("legate={}", fmt::underlying(int_val));
  const auto converted    = legate::detail::convert_log_levels(spec);

  ASSERT_EQ(converted, spec);
}

TEST_F(ArgumentParsingLoggingUnit, MultiConfig)
{
  constexpr auto spec     = std::string_view{"legate=info,dummy_logger=error"};
  constexpr auto expected = std::string_view{"legate=2,dummy_logger=5"};
  const auto converted    = legate::detail::convert_log_levels(spec);

  ASSERT_EQ(converted, expected);
}

TEST_F(ArgumentParsingLoggingUnit, MissingLoggerLevel)
{
  constexpr auto spec = std::string_view{"legate="};

  ASSERT_THAT([&] { static_cast<void>(legate::detail::convert_log_levels(spec)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Unknown logging level '' (from 'legate='), expected one of "
                                     "[all, debug, info, error, none]")));
}

TEST_F(ArgumentParsingLoggingUnit, MissingLoggerName)
{
  constexpr auto spec = std::string_view{"=foo"};

  ASSERT_THAT(
    [&] { static_cast<void>(legate::detail::convert_log_levels(spec)); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
      "Invalid logger specification '=foo', has no logger name. Expected 'logger_name=value'.")));
}

TEST_F(ArgumentParsingLoggingUnit, InvalidLoggerLevel)
{
  constexpr auto spec = std::string_view{"legate=invalid_level"};

  ASSERT_THAT(
    [&] { static_cast<void>(legate::detail::convert_log_levels(spec)); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
      "Unknown logging level 'invalid_level' (from 'legate=invalid_level'), expected one of "
      "[all, debug, info, error, none]")));
}

}  // namespace argument_parsing_logging_test
