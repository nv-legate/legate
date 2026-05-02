/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/string_utils.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utilities/utilities.h>
#include <vector>

namespace string_utils_test {

namespace {

using StringUtilsUnit = DefaultFixture;

}  // namespace

TEST_F(StringUtilsUnit, SplitUnterminatedDoubleQuoteThrows)
{
  constexpr std::string_view input = "\"unterminated";

  ASSERT_THAT(
    [&] { (void)legate::detail::string_split(input); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("Unterminated quote")));
}

TEST_F(StringUtilsUnit, SplitUnterminatedSingleQuoteThrows)
{
  constexpr std::string_view input = "'no closing tick";

  ASSERT_THAT(
    [&] { (void)legate::detail::string_split(input); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("Unterminated quote")));
}

TEST_F(StringUtilsUnit, SplitTerminatedQuotePreservesSpaces)
{
  constexpr std::string_view input        = "\"hello world\" bye";
  const std::vector<std::string> expected = {"hello world", "bye"};

  ASSERT_EQ(legate::detail::string_split<std::string>(input), expected);
}

TEST_F(StringUtilsUnit, SplitEmptyQuotedStringSkipped)
{
  constexpr std::string_view input        = "\"\" keep";
  const std::vector<std::string> expected = {"keep"};

  ASSERT_EQ(legate::detail::string_split<std::string>(input), expected);
}

}  // namespace string_utils_test
