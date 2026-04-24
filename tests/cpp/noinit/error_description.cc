/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/error.h>

#include <cpptrace/basic.hpp>

#include <gtest/gtest.h>

#include <string>
#include <utilities/utilities.h>
#include <vector>

namespace error_description_test {

using ErrorDescriptionUnit = DefaultFixture;

TEST_F(ErrorDescriptionUnit, VectorConstructor)
{
  const std::vector<std::string> lines{"line1", "line2", "line3"};
  const legate::detail::ErrorDescription desc{lines, cpptrace::stacktrace{}};

  ASSERT_EQ(desc.message_lines, lines);
  ASSERT_TRUE(desc.trace.empty());
}

TEST_F(ErrorDescriptionUnit, StringConstructor)
{
  const std::string message = "single-line error";
  const legate::detail::ErrorDescription desc{message};

  ASSERT_EQ(desc.message_lines.size(), 1);
  ASSERT_EQ(desc.message_lines.front(), message);
  ASSERT_TRUE(desc.trace.empty());
}

TEST_F(ErrorDescriptionUnit, CapturesCurrentStacktrace)
{
  const legate::detail::ErrorDescription desc{std::string{"err"}, cpptrace::stacktrace::current()};

  ASSERT_FALSE(desc.trace.empty());
}

}  // namespace error_description_test
