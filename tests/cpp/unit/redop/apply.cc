/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/redop/common.h>

namespace redop_test {

namespace {

// ==========================================================================================
// Test apply operations for basic reductions (EXCLUSIVE = true)
// ==========================================================================================

template <typename T>
class SumReductionApplyTest : public RedopUnit {};

template <typename T>
class ProdReductionApplyTest : public RedopUnit {};

template <typename T>
class MaxReductionApplyTest : public RedopUnit {};

template <typename T>
class MinReductionApplyTest : public RedopUnit {};

TYPED_TEST_SUITE(SumReductionApplyTest, NumericTypes, );
TYPED_TEST_SUITE(ProdReductionApplyTest, NumericTypes, );
TYPED_TEST_SUITE(MaxReductionApplyTest, NumericTypes, );
TYPED_TEST_SUITE(MinReductionApplyTest, NumericTypes, );

TYPED_TEST(SumReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::SumReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(3);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(8));

  // Test with identity
  T val = static_cast<T>(9);
  Reduction::template apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(9));
}

TYPED_TEST(ProdReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::ProdReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(3);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(15));

  // Test with identity
  T val = static_cast<T>(7);
  Reduction::template apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(7));
}

TYPED_TEST(MaxReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::MaxReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(8);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(8));

  lhs          = static_cast<T>(7);
  const T rhs2 = static_cast<T>(3);
  Reduction::template apply<true>(lhs, rhs2);
  ASSERT_EQ(lhs, static_cast<T>(7));
}

TYPED_TEST(MinReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::MinReduction<T>;

  T lhs       = static_cast<T>(5);
  const T rhs = static_cast<T>(8);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(5));

  lhs          = static_cast<T>(6);
  const T rhs2 = static_cast<T>(3);
  Reduction::template apply<true>(lhs, rhs2);
  ASSERT_EQ(lhs, static_cast<T>(3));
}

// ==========================================================================================
// Test apply operations for bool type (separate tests as bool is conceptually not a numeric type)
// ==========================================================================================

TEST_F(RedopUnit, BoolSumReductionApply)
{
  using Reduction = legate::SumReduction<bool>;

  // For bool, sum is logical OR (0+0=0, 0+1=1, 1+0=1, 1+1=1 due to bool conversion)
  bool lhs       = false;
  const bool rhs = false;
  Reduction::apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, false);

  lhs             = false;
  const bool rhs2 = true;
  Reduction::apply<true>(lhs, rhs2);
  ASSERT_EQ(lhs, true);

  lhs             = true;
  const bool rhs3 = true;
  Reduction::apply<true>(lhs, rhs3);
  ASSERT_EQ(lhs, true);

  // Test with identity
  bool val = true;
  Reduction::apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, true);
}

TEST_F(RedopUnit, BoolProdReductionApply)
{
  using Reduction = legate::ProdReduction<bool>;

  // For bool, multiplication is AND
  bool lhs       = true;
  const bool rhs = true;
  Reduction::apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, true);

  lhs             = true;
  const bool rhs2 = false;
  Reduction::apply<true>(lhs, rhs2);
  ASSERT_EQ(lhs, false);

  lhs             = false;
  const bool rhs3 = true;
  Reduction::apply<true>(lhs, rhs3);
  ASSERT_EQ(lhs, false);

  // Test with identity
  bool val = false;
  Reduction::apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, false);
}

TEST_F(RedopUnit, BoolMaxReductionApply)
{
  using Reduction = legate::MaxReduction<bool>;

  bool lhs       = false;
  const bool rhs = true;
  Reduction::apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, true);

  lhs             = true;
  const bool rhs2 = false;
  Reduction::apply<true>(lhs, rhs2);
  ASSERT_EQ(lhs, true);

  lhs             = false;
  const bool rhs3 = false;
  Reduction::apply<true>(lhs, rhs3);
  ASSERT_EQ(lhs, false);
}

TEST_F(RedopUnit, BoolMinReductionApply)
{
  using Reduction = legate::MinReduction<bool>;

  bool lhs       = true;
  const bool rhs = false;
  Reduction::apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, false);

  lhs             = false;
  const bool rhs2 = true;
  Reduction::apply<true>(lhs, rhs2);
  ASSERT_EQ(lhs, false);

  lhs             = true;
  const bool rhs3 = true;
  Reduction::apply<true>(lhs, rhs3);
  ASSERT_EQ(lhs, true);
}

// ==========================================================================================
// Test apply operations for bitwise reductions (EXCLUSIVE = true)
// ==========================================================================================

template <typename T>
class OrReductionApplyTest : public RedopUnit {};

template <typename T>
class AndReductionApplyTest : public RedopUnit {};

template <typename T>
class XorReductionApplyTest : public RedopUnit {};

TYPED_TEST_SUITE(OrReductionApplyTest, IntegerTypes, );
TYPED_TEST_SUITE(AndReductionApplyTest, IntegerTypes, );
TYPED_TEST_SUITE(XorReductionApplyTest, IntegerTypes, );

TYPED_TEST(OrReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::OrReduction<T>;

  T lhs       = static_cast<T>(K_BITS_1010);
  const T rhs = static_cast<T>(K_BITS_0101);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(K_BITS_1111));

  // Test with identity
  T val = static_cast<T>(K_BITS_1100);
  Reduction::template apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_BITS_1100));
}

TYPED_TEST(AndReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::AndReduction<T>;

  T lhs       = static_cast<T>(K_BITS_1111);
  const T rhs = static_cast<T>(K_BITS_0101);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(K_BITS_0101));

  // Test with identity
  T val = static_cast<T>(K_BITS_1100);
  Reduction::template apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_BITS_1100));
}

TYPED_TEST(XorReductionApplyTest, Apply)
{
  using T         = TypeParam;
  using Reduction = legate::XORReduction<T>;

  T lhs       = static_cast<T>(K_BITS_1010);
  const T rhs = static_cast<T>(K_BITS_1100);

  Reduction::template apply<true>(lhs, rhs);
  ASSERT_EQ(lhs, static_cast<T>(K_BITS_0110));

  // Test with identity
  T val = static_cast<T>(K_BITS_1010);
  Reduction::template apply<true>(val, Reduction::identity);
  ASSERT_EQ(val, static_cast<T>(K_BITS_1010));
}

// ==========================================================================================
// Test apply operations with EXCLUSIVE = false (atomic path)
// ==========================================================================================

TEST_F(RedopUnit, SumReductionApplyAtomic)
{
  using Reduction = legate::SumReduction<int>;

  int lhs       = 5;
  const int rhs = 3;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, 8);
}

TEST_F(RedopUnit, ProdReductionApplyAtomic)
{
  using Reduction = legate::ProdReduction<int>;

  int lhs       = 5;
  const int rhs = 3;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, 15);
}

TEST_F(RedopUnit, MaxReductionApplyAtomic)
{
  using Reduction = legate::MaxReduction<int>;

  int lhs       = 5;
  const int rhs = 8;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, 8);

  lhs            = K_INT_10;
  const int rhs2 = 3;
  Reduction::apply<false>(lhs, rhs2);
  ASSERT_EQ(lhs, K_INT_10);
}

TEST_F(RedopUnit, MinReductionApplyAtomic)
{
  using Reduction = legate::MinReduction<int>;

  int lhs       = 5;
  const int rhs = 8;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, 5);

  lhs            = K_INT_10;
  const int rhs2 = 3;
  Reduction::apply<false>(lhs, rhs2);
  ASSERT_EQ(lhs, 3);
}

TEST_F(RedopUnit, OrReductionApplyAtomic)
{
  using Reduction = legate::OrReduction<int>;

  int lhs       = K_BITS_1010;
  const int rhs = K_BITS_0101;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, K_BITS_1111);
}

TEST_F(RedopUnit, AndReductionApplyAtomic)
{
  using Reduction = legate::AndReduction<int>;

  int lhs       = K_BITS_1111;
  const int rhs = K_BITS_0101;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, K_BITS_0101);
}

TEST_F(RedopUnit, XorReductionApplyAtomic)
{
  using Reduction = legate::XORReduction<int>;

  int lhs       = K_BITS_1010;
  const int rhs = K_BITS_1100;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_EQ(lhs, K_BITS_0110);
}

// ==========================================================================================
// Test apply with floating point types (atomic path)
// ==========================================================================================

TEST_F(RedopUnit, FloatSumReductionApplyAtomic)
{
  using Reduction = legate::SumReduction<float>;

  float lhs       = K_FLOAT_5_5;
  const float rhs = 3.5F;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs, 9.0F);
}

TEST_F(RedopUnit, FloatProdReductionApplyAtomic)
{
  using Reduction = legate::ProdReduction<float>;

  float lhs       = K_FLOAT_2_0;
  const float rhs = 3.0F;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs, 6.0F);
}

TEST_F(RedopUnit, FloatMaxReductionApplyAtomic)
{
  using Reduction = legate::MaxReduction<float>;

  float lhs       = K_FLOAT_5_5;
  const float rhs = 8.5F;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs, 8.5F);
}

TEST_F(RedopUnit, FloatMinReductionApplyAtomic)
{
  using Reduction = legate::MinReduction<float>;

  float lhs       = K_FLOAT_5_5;
  const float rhs = 3.5F;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs, 3.5F);
}

TEST_F(RedopUnit, DoubleSumReductionApplyAtomic)
{
  using Reduction = legate::SumReduction<double>;

  double lhs       = K_DOUBLE_5_5;
  const double rhs = 3.5;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_DOUBLE_EQ(lhs, 9.0);
}

TEST_F(RedopUnit, DoubleMaxReductionApplyAtomic)
{
  using Reduction = legate::MaxReduction<double>;

  double lhs       = K_DOUBLE_5_5;
  const double rhs = 8.5;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_DOUBLE_EQ(lhs, 8.5);
}

TEST_F(RedopUnit, DoubleMinReductionApplyAtomic)
{
  using Reduction = legate::MinReduction<double>;

  double lhs       = K_DOUBLE_5_5;
  const double rhs = 3.5;

  Reduction::apply<false>(lhs, rhs);
  ASSERT_DOUBLE_EQ(lhs, 3.5);
}

}  // namespace

}  // namespace redop_test
