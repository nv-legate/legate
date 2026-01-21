/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/redop/common.h>

namespace redop_test {

namespace {

// ==========================================================================================
// Test fold operations for basic reductions
// ==========================================================================================

template <typename T>
class SumReductionFoldTest : public RedopUnit {};

template <typename T>
class ProdReductionFoldTest : public RedopUnit {};

template <typename T>
class MaxReductionFoldTest : public RedopUnit {};

template <typename T>
class MinReductionFoldTest : public RedopUnit {};

TYPED_TEST_SUITE(SumReductionFoldTest, NumericTypes, );
TYPED_TEST_SUITE(ProdReductionFoldTest, NumericTypes, );
TYPED_TEST_SUITE(MaxReductionFoldTest, NumericTypes, );
TYPED_TEST_SUITE(MinReductionFoldTest, NumericTypes, );

TYPED_TEST(SumReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::SumReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(3);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(8));

  // Test with identity
  T val = static_cast<T>(K_INT_10);
  Reduction::template fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_INT_10));
}

TYPED_TEST(ProdReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::ProdReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(3);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(15));

  // Test with identity
  T val = static_cast<T>(7);
  Reduction::template fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(7));
}

TYPED_TEST(MaxReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::MaxReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(8);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(8));

  lhs          = static_cast<T>(K_INT_10);
  const T rhs2 = static_cast<T>(3);
  Reduction::template fold<true>(lhs, rhs2);
  ASSERT_EQ(lhs, static_cast<T>(K_INT_10));
}

TYPED_TEST(MinReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::MinReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(8);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(5));

  lhs          = static_cast<T>(K_INT_10);
  const T rhs2 = static_cast<T>(3);
  Reduction::template fold<true>(lhs, rhs2);
  ASSERT_EQ(lhs, static_cast<T>(3));
}

// ==========================================================================================
// Test fold operations for bool type (separate tests as bool is conceptually not a numeric type)
// ==========================================================================================

TEST_F(RedopUnit, BoolSumReductionFold)
{
  using Reduction = legate::SumReduction<bool>;

  // For bool, sum is logical OR
  bool lhs       = false;
  const bool rhs = false;
  Reduction::fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, false);

  lhs             = false;
  const bool rhs2 = true;
  Reduction::fold<true>(lhs, rhs2);
  ASSERT_EQ(lhs, true);

  lhs             = true;
  const bool rhs3 = true;
  Reduction::fold<true>(lhs, rhs3);
  ASSERT_EQ(lhs, true);

  // Test with identity
  bool val = true;
  Reduction::fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, true);
}

TEST_F(RedopUnit, BoolProdReductionFold)
{
  using Reduction = legate::ProdReduction<bool>;

  // For bool, multiplication is AND
  bool lhs       = true;
  const bool rhs = true;
  Reduction::fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, true);

  lhs             = true;
  const bool rhs2 = false;
  Reduction::fold<true>(lhs, rhs2);
  ASSERT_EQ(lhs, false);

  lhs             = false;
  const bool rhs3 = true;
  Reduction::fold<true>(lhs, rhs3);
  ASSERT_EQ(lhs, false);

  // Test with identity
  bool val = false;
  Reduction::fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, false);
}

TEST_F(RedopUnit, BoolMaxReductionFold)
{
  using Reduction = legate::MaxReduction<bool>;

  bool lhs       = false;
  const bool rhs = true;
  Reduction::fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, true);

  lhs             = true;
  const bool rhs2 = false;
  Reduction::fold<true>(lhs, rhs2);
  ASSERT_EQ(lhs, true);

  lhs             = false;
  const bool rhs3 = false;
  Reduction::fold<true>(lhs, rhs3);
  ASSERT_EQ(lhs, false);
}

TEST_F(RedopUnit, BoolMinReductionFold)
{
  using Reduction = legate::MinReduction<bool>;

  bool lhs       = true;
  const bool rhs = false;
  Reduction::fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, false);

  lhs             = false;
  const bool rhs2 = true;
  Reduction::fold<true>(lhs, rhs2);
  ASSERT_EQ(lhs, false);

  lhs             = true;
  const bool rhs3 = true;
  Reduction::fold<true>(lhs, rhs3);
  ASSERT_EQ(lhs, true);
}

// ==========================================================================================
// Test fold operations for bitwise reductions
// ==========================================================================================

template <typename T>
class OrReductionFoldTest : public RedopUnit {};

template <typename T>
class AndReductionFoldTest : public RedopUnit {};

template <typename T>
class XorReductionFoldTest : public RedopUnit {};

TYPED_TEST_SUITE(OrReductionFoldTest, IntegerTypes, );
TYPED_TEST_SUITE(AndReductionFoldTest, IntegerTypes, );
TYPED_TEST_SUITE(XorReductionFoldTest, IntegerTypes, );

TYPED_TEST(OrReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::OrReduction<T>;

  T lhs       = static_cast<T>(K_BITS_1010);
  const T rhs = static_cast<T>(K_BITS_0101);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(K_BITS_1111));

  // Test with identity
  T val = static_cast<T>(K_BITS_1100);
  Reduction::template fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_BITS_1100));
}

TYPED_TEST(AndReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::AndReduction<T>;

  T lhs       = static_cast<T>(K_BITS_1111);
  const T rhs = static_cast<T>(K_BITS_0101);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(K_BITS_0101));

  // Test with identity
  T val = static_cast<T>(K_BITS_1100);
  Reduction::template fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_BITS_1100));
}

TYPED_TEST(XorReductionFoldTest, Fold)
{
  using T         = TypeParam;
  using Reduction = legate::XORReduction<T>;

  T lhs       = static_cast<T>(K_BITS_1010);
  const T rhs = static_cast<T>(K_BITS_1100);

  Reduction::template fold<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(K_BITS_0110));

  // Test with identity
  T val = static_cast<T>(K_BITS_1010);
  Reduction::template fold<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_BITS_1010));
}

// ==========================================================================================
// Test type traits
// ==========================================================================================

TEST_F(RedopUnit, ReductionTypeTraits)
{
  // Verify that reduction types have the expected value_type
  static_assert(std::is_same_v<legate::SumReduction<int>::LHS, int>);
  static_assert(std::is_same_v<legate::SumReduction<int>::RHS, int>);

  static_assert(std::is_same_v<legate::ProdReduction<float>::LHS, float>);
  static_assert(std::is_same_v<legate::ProdReduction<float>::RHS, float>);

  static_assert(std::is_same_v<legate::MaxReduction<double>::LHS, double>);
  static_assert(std::is_same_v<legate::MaxReduction<double>::RHS, double>);

  static_assert(std::is_same_v<legate::MinReduction<std::int64_t>::LHS, std::int64_t>);
  static_assert(std::is_same_v<legate::MinReduction<std::int64_t>::RHS, std::int64_t>);

  static_assert(std::is_same_v<legate::OrReduction<std::uint32_t>::LHS, std::uint32_t>);
  static_assert(std::is_same_v<legate::AndReduction<std::uint32_t>::LHS, std::uint32_t>);
  static_assert(std::is_same_v<legate::XORReduction<std::uint32_t>::LHS, std::uint32_t>);
}

}  // namespace

}  // namespace redop_test
