/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/config_legion.h>
#include <legate/runtime/detail/argument_parsing/parse.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_compose_legion_default_args {

class ComposeLegionDefaultArgsUnit : public DefaultFixture {};

#define UNSET_ENV_VAR(__var_name__) \
  const auto __var_name__ = legate::test::Environment::TemporaryEnvVar { #__var_name__, nullptr }

TEST_F(ComposeLegionDefaultArgsUnit, Basic)
{
  UNSET_ENV_VAR(LEGION_DEFAULT_ARGS);
  UNSET_ENV_VAR(LEGATE_LOG_MAPPING);
  UNSET_ENV_VAR(LEGATE_LOG_PARTITIONING);

  const auto parsed = legate::detail::parse_args({"parsed", "--omps", "0", "--numamem", "0"});
  const auto legion_default_args = legate::detail::compose_legion_default_args(parsed);
  std::string expected           = "-lg:local 0 ";

  if (LEGATE_DEFINED(LEGATE_HAS_ASAN)) {
    expected += "-ll:force_kthreads ";
  }

  ASSERT_EQ(legion_default_args, expected);
}

TEST_F(ComposeLegionDefaultArgsUnit, WithFlags)
{
  UNSET_ENV_VAR(LEGION_DEFAULT_ARGS);
  UNSET_ENV_VAR(LEGATE_LOG_MAPPING);
  UNSET_ENV_VAR(LEGATE_LOG_PARTITIONING);

  const auto parsed = legate::detail::parse_args({"parsed",
                                                  "--omps",
                                                  "1",
                                                  "--numamem",
                                                  "0",
                                                  "--spy",
                                                  "--profile",
                                                  "--logging",
                                                  "foo=info,baz=debug",
                                                  "--log-to-file",
                                                  "--freeze-on-error"});

  const auto legion_default_args = legate::detail::compose_legion_default_args(parsed);

  const auto expected = fmt::format(
    "-lg:local 0 "
    "-ll:onuma 0 "
    "-lg:spy "
    "-lg:prof 1 "
    "-lg:prof_logfile \"{}\" "
    "-level foo=2,baz=1,legion_spy=2,legion_prof=2 "
    "-logfile \"{}\" "
    "-errlevel 4 "
    "-ll:force_kthreads ",
    std::filesystem::current_path() / "legate_%.prof",
    std::filesystem::current_path() / "legate_%.log");

  ASSERT_EQ(legion_default_args, expected);
}

TEST_F(ComposeLegionDefaultArgsUnit, WithFlagsAndDefaultArgs)
{
  UNSET_ENV_VAR(LEGATE_LOG_MAPPING);
  UNSET_ENV_VAR(LEGATE_LOG_PARTITIONING);

  const auto LEGION_DEFAULT_ARGS = legate::test::Environment::TemporaryEnvVar{
    "LEGION_DEFAULT_ARGS", "--some --default --legion --args", /* overwrite */ true};

  const auto parsed = legate::detail::parse_args({"parsed",
                                                  "--omps",
                                                  "1",
                                                  "--numamem",
                                                  "0",
                                                  "--spy",
                                                  "--profile",
                                                  "--logging",
                                                  "foo=info,baz=debug",
                                                  "--log-to-file",
                                                  "--freeze-on-error"});

  const auto legion_default_args = legate::detail::compose_legion_default_args(parsed);

  const auto expected = fmt::format(
    "-lg:local 0 "
    "-ll:onuma 0 "
    "-lg:spy "
    "-lg:prof 1 "
    "-lg:prof_logfile \"{}\" "
    "-level foo=2,baz=1,legion_spy=2,legion_prof=2 "
    "-logfile \"{}\" "
    "-errlevel 4 "
    "-ll:force_kthreads "
    " --some --default --legion --args",
    std::filesystem::current_path() / "legate_%.prof",
    std::filesystem::current_path() / "legate_%.log");

  ASSERT_EQ(legion_default_args, expected);
}

class ConfigureLegionUnit : public DefaultFixture {};

TEST_F(ConfigureLegionUnit, Basic)
{
  UNSET_ENV_VAR(LEGION_DEFAULT_ARGS);
  UNSET_ENV_VAR(LEGION_FREEZE_ON_ERROR);
  UNSET_ENV_VAR(LEGATE_LOG_MAPPING);
  UNSET_ENV_VAR(LEGATE_LOG_PARTITIONING);

  const auto parsed = legate::detail::parse_args({"parsed", "--omps", "0", "--numamem", "0"});

  legate::detail::configure_legion(parsed);

  std::string expected = "-lg:local 0 ";

  if (LEGATE_DEFINED(LEGATE_HAS_ASAN)) {
    expected += "-ll:force_kthreads ";
  }

  const auto value = legate::detail::LEGION_DEFAULT_ARGS.get();

  ASSERT_THAT(value, ::testing::Optional(expected));

  const auto freeze_on_error =
    legate::detail::EnvironmentVariable<bool>{"LEGION_FREEZE_ON_ERROR"}.get();

  ASSERT_EQ(freeze_on_error, std::nullopt);
}

TEST_F(ConfigureLegionUnit, WithFlags)
{
  UNSET_ENV_VAR(LEGION_DEFAULT_ARGS);
  UNSET_ENV_VAR(LEGION_FREEZE_ON_ERROR);
  UNSET_ENV_VAR(LEGATE_LOG_MAPPING);
  UNSET_ENV_VAR(LEGATE_LOG_PARTITIONING);

  const auto parsed = legate::detail::parse_args({"parsed",
                                                  "--omps",
                                                  "1",
                                                  "--numamem",
                                                  "0",
                                                  "--spy",
                                                  "--profile",
                                                  "--logging",
                                                  "foo=info,baz=debug",
                                                  "--log-to-file",
                                                  "--freeze-on-error"});

  legate::detail::configure_legion(parsed);

  const auto expected = fmt::format(
    "-lg:local 0 "
    "-ll:onuma 0 "
    "-lg:spy "
    "-lg:prof 1 "
    "-lg:prof_logfile \"{}\" "
    "-level foo=2,baz=1,legion_spy=2,legion_prof=2 "
    "-logfile \"{}\" "
    "-errlevel 4 "
    "-ll:force_kthreads ",
    std::filesystem::current_path() / "legate_%.prof",
    std::filesystem::current_path() / "legate_%.log");

  const auto value = legate::detail::LEGION_DEFAULT_ARGS.get();

  ASSERT_THAT(value, ::testing::Optional(expected));

  const auto freeze_on_error =
    legate::detail::EnvironmentVariable<bool>{"LEGION_FREEZE_ON_ERROR"}.get();

  ASSERT_THAT(freeze_on_error, ::testing::Optional(true));
}

TEST_F(ConfigureLegionUnit, WithFlagsAndDefaultArgs)
{
  UNSET_ENV_VAR(LEGION_FREEZE_ON_ERROR);
  UNSET_ENV_VAR(LEGATE_LOG_MAPPING);
  UNSET_ENV_VAR(LEGATE_LOG_PARTITIONING);

  const auto LEGION_DEFAULT_ARGS = legate::test::Environment::TemporaryEnvVar{
    "LEGION_DEFAULT_ARGS", "--some --default --legion --args", /* overwrite */ true};

  const auto parsed = legate::detail::parse_args({"parsed",
                                                  "--omps",
                                                  "1",
                                                  "--numamem",
                                                  "0",
                                                  "--spy",
                                                  "--profile",
                                                  "--logging",
                                                  "foo=info,baz=debug",
                                                  "--log-to-file",
                                                  "--freeze-on-error"});

  legate::detail::configure_legion(parsed);

  const auto expected = fmt::format(
    "-lg:local 0 "
    "-ll:onuma 0 "
    "-lg:spy "
    "-lg:prof 1 "
    "-lg:prof_logfile \"{}\" "
    "-level foo=2,baz=1,legion_spy=2,legion_prof=2 "
    "-logfile \"{}\" "
    "-errlevel 4 "
    "-ll:force_kthreads "
    " --some --default --legion --args",
    std::filesystem::current_path() / "legate_%.prof",
    std::filesystem::current_path() / "legate_%.log");

  const auto value = legate::detail::LEGION_DEFAULT_ARGS.get();

  ASSERT_THAT(value, ::testing::Optional(expected));

  const auto freeze_on_error =
    legate::detail::EnvironmentVariable<bool>{"LEGION_FREEZE_ON_ERROR"}.get();

  ASSERT_THAT(freeze_on_error, ::testing::Optional(true));
}

}  // namespace test_compose_legion_default_args
