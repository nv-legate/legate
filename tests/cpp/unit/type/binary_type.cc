/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace binary_type_test {

namespace {

using BinaryTypeUnit = DefaultFixture;

class BinaryTypeTest : public BinaryTypeUnit,
                       public ::testing::WithParamInterface<std::uint32_t> {};

INSTANTIATE_TEST_SUITE_P(BinaryTypeUnit,
                         BinaryTypeTest,
                         ::testing::Values(123, 45, 0, 0xFFFFF),
                         ::testing::PrintToStringParamName{});

}  // namespace

TEST_P(BinaryTypeTest, Basic)
{
  const legate::Type type  = legate::binary_type(GetParam());
  const std::uint32_t size = GetParam();

  ASSERT_EQ(type.code(), legate::Type::Code::BINARY);
  ASSERT_EQ(type.size(), size);
  ASSERT_EQ(type.alignment(), alignof(std::max_align_t));
  ASSERT_FALSE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), fmt::format("binary({})", size));

  // Note: aim to test the copy initialization of Type
  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

TEST_F(BinaryTypeUnit, BinaryTypeBad)
{
  constexpr auto BINARY_TYPE_OVER_SIZE = 0xFFFFF + 1;

  ASSERT_THAT([&]() { static_cast<void>(legate::binary_type(BINARY_TYPE_OVER_SIZE)); },
              testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("Maximum size for opaque binary types is")));
}

}  // namespace binary_type_test
