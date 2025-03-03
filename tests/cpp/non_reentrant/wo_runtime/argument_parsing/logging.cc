/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/logging.h>

#include <legate/runtime/runtime.h>
#include <legate/utilities/detail/env.h>

#include <legion.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace argument_parsing_logging_test {

class ArgumentParsingLoggingUnitFixture : public DefaultFixture {
  void TearDown() override
  {
    if (legate::has_started()) {
      ASSERT_EQ(legate::finish(), 0);
    }
    DefaultFixture::TearDown();
  }
};

class ArgumentParsingLoggingUnit
  : public ArgumentParsingLoggingUnitFixture,
    public ::testing::WithParamInterface<
      std::tuple<std::string_view, Realm::Logger::LoggingLevel, bool>> {};

INSTANTIATE_TEST_SUITE_P(
  ArgumentParsingLoggingUnit,
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
  const auto lvl                       = fmt::format("legate={}", str);

  if (should_handle) {
    const auto expected = fmt::format("legate={}", fmt::underlying(int_val));

    ASSERT_EQ(legate::detail::convert_log_levels(lvl), expected);
  } else {
    ASSERT_THROW(static_cast<void>(legate::detail::convert_log_levels(lvl)), std::invalid_argument);
  }
}

TEST_P(ArgumentParsingLoggingUnit, ConvertLogLevelAllowInt)
{
  auto&& [_, int_val, _2] = GetParam();
  const auto lvl          = fmt::format("legate={}", fmt::underlying(int_val));

  ASSERT_EQ(legate::detail::convert_log_levels(lvl), lvl);
}

TEST_P(ArgumentParsingLoggingUnit, SetLevel)
{
  auto&& [str, int_val, should_handle] = GetParam();
  const auto _                         = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CONFIG, fmt::format("--logging legate={}", str).c_str(), true);

  if (should_handle) {
    legate::start();
    ASSERT_EQ(static_cast<std::int32_t>(legate::detail::log_legate().get_level()), int_val);
    ASSERT_EQ(legate::finish(), 0);
  } else {
    ASSERT_THROW(legate::start(), std::invalid_argument);
  }
}

TEST_P(ArgumentParsingLoggingUnit, SetLevelAllowInt)
{
  auto&& [_, int_val, _2] = GetParam();
  const auto _3           = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CONFIG,
    fmt::format("--logging legate={}", fmt::underlying(int_val)).c_str(),
    true);

  legate::start();
  ASSERT_EQ(static_cast<std::int32_t>(legate::detail::log_legate().get_level()), int_val);
  ASSERT_EQ(legate::finish(), 0);
}

TEST_F(ArgumentParsingLoggingUnit, MultiConfig)
{
  const auto dummy_logger = legate::Logger{"dummy_logger"};
  const auto _            = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CONFIG, "--logging legate=info,dummy_logger=error", true);

  legate::start();
  ASSERT_EQ(static_cast<std::int32_t>(legate::detail::log_legate().get_level()),
            Realm::Logger::LoggingLevel::LEVEL_INFO);
  ASSERT_EQ(static_cast<std::int32_t>(dummy_logger.get_level()),
            Realm::Logger::LoggingLevel::LEVEL_ERROR);
  ASSERT_EQ(legate::finish(), 0);
}

TEST_F(ArgumentParsingLoggingUnit, MissingLoggerLevel)
{
  const auto _ = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CONFIG, "--logging legate=", true);

  ASSERT_THROW(legate::start(), std::invalid_argument);
}

TEST_F(ArgumentParsingLoggingUnit, MissingLoggerName)
{
  const auto _ = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CONFIG, "--logging =foo", true);

  ASSERT_THROW(legate::start(), std::invalid_argument);
}

TEST_F(ArgumentParsingLoggingUnit, InvalidLoggerLevel)
{
  const auto _ = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CONFIG, "--logging legate=invalid_level", true);

  ASSERT_THROW(legate::start(), std::invalid_argument);
}

using ArgumentParsingLoggingUnitDeathTest = ArgumentParsingLoggingUnit;

TEST_F(ArgumentParsingLoggingUnitDeathTest, NoArg)
{
  const auto _ =
    legate::test::Environment::temporary_env_var(legate::detail::LEGATE_CONFIG, "--logging", true);

  ASSERT_EXIT(legate::start(),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              ::testing::HasSubstr("Too few arguments for '--logging'"));
}

}  // namespace argument_parsing_logging_test
