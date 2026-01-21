/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/redop/common.h>

namespace redop_test {

namespace {

// ==========================================================================================
// Test identity values for basic reduction operations
// ==========================================================================================

template <typename T>
class SumReductionIdentityTest : public RedopUnit {};

template <typename T>
class ProdReductionIdentityTest : public RedopUnit {};

template <typename T>
class MaxReductionIdentityTest : public RedopUnit {};

template <typename T>
class MinReductionIdentityTest : public RedopUnit {};

TYPED_TEST_SUITE(SumReductionIdentityTest, NumericTypes, );
TYPED_TEST_SUITE(ProdReductionIdentityTest, NumericTypes, );
TYPED_TEST_SUITE(MaxReductionIdentityTest, NumericTypes, );
TYPED_TEST_SUITE(MinReductionIdentityTest, NumericTypes, );

TYPED_TEST(SumReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::SumReduction<T>;

  // Sum identity should be 0
  ASSERT_EQ(Reduction::identity, static_cast<T>(0));
}

TYPED_TEST(ProdReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::ProdReduction<T>;

  // Prod identity should be 1
  ASSERT_EQ(Reduction::identity, static_cast<T>(1));
}

TYPED_TEST(MaxReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::MaxReduction<T>;

  // Max identity should be the minimum value for the type
  // Note: Legion uses lowest() for floating point types, not -infinity()
  ASSERT_EQ(Reduction::identity, std::numeric_limits<T>::lowest());
}

TYPED_TEST(MinReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::MinReduction<T>;

  // Min identity should be the maximum value for the type
  // Note: Legion uses max() for floating point types, not infinity()
  ASSERT_EQ(Reduction::identity, std::numeric_limits<T>::max());
}

// ==========================================================================================
// Test identity values for bool type (separate tests as bool is conceptually not a numeric type)
// ==========================================================================================

TEST_F(RedopUnit, BoolSumReductionIdentity)
{
  using Reduction = legate::SumReduction<bool>;

  // Sum identity should be false (0)
  ASSERT_EQ(Reduction::identity, false);
}

TEST_F(RedopUnit, BoolProdReductionIdentity)
{
  using Reduction = legate::ProdReduction<bool>;

  // Prod identity should be true (1)
  ASSERT_EQ(Reduction::identity, true);
}

TEST_F(RedopUnit, BoolMaxReductionIdentity)
{
  using Reduction = legate::MaxReduction<bool>;

  // Max identity should be false (minimum value for bool)
  ASSERT_EQ(Reduction::identity, false);
}

TEST_F(RedopUnit, BoolMinReductionIdentity)
{
  using Reduction = legate::MinReduction<bool>;

  // Min identity should be true (maximum value for bool)
  ASSERT_EQ(Reduction::identity, true);
}

// ==========================================================================================
// Test identity values for bitwise reduction operations
// ==========================================================================================

template <typename T>
class OrReductionIdentityTest : public RedopUnit {};

template <typename T>
class AndReductionIdentityTest : public RedopUnit {};

template <typename T>
class XorReductionIdentityTest : public RedopUnit {};

TYPED_TEST_SUITE(OrReductionIdentityTest, IntegerTypes, );
TYPED_TEST_SUITE(AndReductionIdentityTest, IntegerTypes, );
TYPED_TEST_SUITE(XorReductionIdentityTest, IntegerTypes, );

TYPED_TEST(OrReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::OrReduction<T>;

  // Or identity should be 0 (all bits clear)
  ASSERT_EQ(Reduction::identity, static_cast<T>(0));
}

TYPED_TEST(AndReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::AndReduction<T>;

  // And identity should satisfy: x & identity == x for small values
  // Note: Legion uses truncated identity (16-bit or 32-bit) for larger types
  const T test_val = static_cast<T>(K_BITS_00FF);
  ASSERT_EQ(test_val & Reduction::identity, test_val);

  // Identity should not be zero
  ASSERT_NE(Reduction::identity, static_cast<T>(0));
}

TYPED_TEST(XorReductionIdentityTest, Identity)
{
  using T         = TypeParam;
  using Reduction = legate::XORReduction<T>;

  // XOR identity should be 0
  ASSERT_EQ(Reduction::identity, static_cast<T>(0));
}

}  // namespace

}  // namespace redop_test
