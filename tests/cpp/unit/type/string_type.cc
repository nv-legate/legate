/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace string_type_test {

namespace {

using StringTypeUnit = DefaultFixture;

}  // namespace

TEST_F(StringTypeUnit, StringType)
{
  const legate::Type type = legate::string_type();

  ASSERT_EQ(type.code(), legate::Type::Code::STRING);
  ASSERT_THAT([&]() { static_cast<void>(type.size()); },
              testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Size of a variable size type is undefined")));
  ASSERT_EQ(type.alignment(), alignof(std::max_align_t));
  ASSERT_TRUE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), "string");

  // Note: aim to test the copy initialization of Type
  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

}  // namespace string_type_test
