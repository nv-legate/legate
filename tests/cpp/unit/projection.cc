/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/projection.h>

#include <gtest/gtest.h>

#include <sstream>

namespace {

// NOLINTBEGIN(readability-magic-numbers)

TEST(SymbolicExpr, ToStringFormatsWeightedExpression)
{
  const auto expr = legate::dimension(2) * 3;

  ASSERT_EQ(expr.to_string(), "3*COORD2");
}

TEST(SymbolicExpr, ToStringFormatsOffsets)
{
  ASSERT_EQ((legate::dimension(1) + 4).to_string(), "COORD1+4");
  ASSERT_EQ((legate::dimension(1) + 0).to_string(), "COORD1");
  ASSERT_EQ((legate::dimension(1) + -4).to_string(), "COORD1-4");
}

TEST(SymbolicExpr, ToStringFormatsConstants)
{
  ASSERT_EQ(legate::constant(4).to_string(), "4");
  ASSERT_EQ(legate::constant(0).to_string(), "0");
  ASSERT_EQ(legate::constant(-4).to_string(), "-4");
}

TEST(SymbolicExpr, StreamInsertionFormatsExpression)
{
  const auto expr = legate::dimension(2) * 3;
  auto out        = std::ostringstream{};

  auto& returned = out << expr;

  ASSERT_EQ(&returned, &out);
  ASSERT_EQ(out.str(), "3*COORD2");
}

TEST(SymbolicExpr, IsIdentityChecksDimensionWeightAndOffset)
{
  constexpr auto dim       = std::uint32_t{2};
  constexpr auto other_dim = std::uint32_t{1};
  const auto identity      = legate::dimension(dim);
  const auto scaled        = identity * 3;
  const auto shifted       = identity + 1;

  ASSERT_TRUE(identity.is_identity(dim));
  ASSERT_FALSE(identity.is_identity(other_dim));
  ASSERT_FALSE(scaled.is_identity(dim));
  ASSERT_FALSE(shifted.is_identity(dim));
}

TEST(SymbolicExpr, OperatorEqualComparesDimensionWeightAndOffset)
{
  constexpr auto dim       = std::uint32_t{2};
  constexpr auto other_dim = std::uint32_t{1};
  const auto expr          = legate::dimension(dim) * 3 + 4;
  const auto same          = legate::dimension(dim) * 3 + 4;
  const auto different_dim = legate::dimension(other_dim) * 3 + 4;
  const auto scaled        = legate::dimension(dim) * 2 + 4;
  const auto shifted       = legate::dimension(dim) * 3 + 5;

  ASSERT_TRUE(expr == same);
  ASSERT_FALSE(expr == different_dim);
  ASSERT_FALSE(expr == scaled);
  ASSERT_FALSE(expr == shifted);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace
