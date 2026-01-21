/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/redop/common.h>

namespace redop_test {

namespace {

// ==========================================================================================
// Test Half type reductions - Identity
// ==========================================================================================

TEST_F(RedopUnit, HalfSumReductionIdentity)
{
  using Reduction = legate::SumReduction<legate::Half>;

  // Sum identity should be 0
  ASSERT_EQ(static_cast<float>(Reduction::identity), 0.0F);
}

TEST_F(RedopUnit, HalfProdReductionIdentity)
{
  using Reduction = legate::ProdReduction<legate::Half>;

  // Prod identity should be 1
  ASSERT_EQ(static_cast<float>(Reduction::identity), 1.0F);
}

TEST_F(RedopUnit, HalfMaxReductionIdentity)
{
  using Reduction = legate::MaxReduction<legate::Half>;

  // Max identity should be the minimum half value
  ASSERT_LT(static_cast<float>(Reduction::identity), -60000.0F);
}

TEST_F(RedopUnit, HalfMinReductionIdentity)
{
  using Reduction = legate::MinReduction<legate::Half>;

  // Min identity should be the maximum half value
  ASSERT_GT(static_cast<float>(Reduction::identity), 60000.0F);
}

// ==========================================================================================
// Test Half type reductions - Fold
// ==========================================================================================

TEST_F(RedopUnit, HalfSumReductionFold)
{
  using Reduction = legate::SumReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);
}

TEST_F(RedopUnit, HalfProdReductionFold)
{
  using Reduction = legate::ProdReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 15.0F);
}

TEST_F(RedopUnit, HalfMaxReductionFold)
{
  using Reduction = legate::MaxReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{8.0F};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);
}

TEST_F(RedopUnit, HalfMinReductionFold)
{
  using Reduction = legate::MinReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 3.0F);
}

// ==========================================================================================
// Test Half type reductions - Apply
// ==========================================================================================

TEST_F(RedopUnit, HalfSumReductionApply)
{
  using Reduction = legate::SumReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);
}

TEST_F(RedopUnit, HalfProdReductionApply)
{
  using Reduction = legate::ProdReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 15.0F);
}

TEST_F(RedopUnit, HalfMaxReductionApply)
{
  using Reduction = legate::MaxReduction<legate::Half>;

  // This explicitly calls Maximum::operator()<__half, __half>
  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{8.0F};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);

  // Test the other direction to ensure both branches are covered
  lhs = legate::Half{K_FLOAT_10_0};
  const legate::Half rhs2{3.0F};
  Reduction::apply<true>(lhs, rhs2);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 10.0F);
}

TEST_F(RedopUnit, HalfMinReductionApply)
{
  using Reduction = legate::MinReduction<legate::Half>;

  // This explicitly calls Minimum::operator()<__half, __half>
  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 3.0F);

  // Test the other direction to ensure both branches are covered
  lhs = legate::Half{K_FLOAT_2_0};
  const legate::Half rhs2{7.0F};
  Reduction::apply<true>(lhs, rhs2);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 2.0F);
}

// ==========================================================================================
// Test Half type reductions - Atomic (EXCLUSIVE = false)
// These tests cover the AtomicWrapper code paths for Half type
// ==========================================================================================

TEST_F(RedopUnit, HalfMaxReductionFoldAtomic)
{
  using Reduction = legate::MaxReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{8.0F};

  // EXCLUSIVE = false triggers AtomicWrapper<Maximum>::operator()<__half>
  Reduction::fold<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);

  // Test the other direction
  lhs = legate::Half{K_FLOAT_10_0};
  const legate::Half rhs2{3.0F};
  Reduction::fold<false>(lhs, rhs2);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 10.0F);
}

TEST_F(RedopUnit, HalfMinReductionFoldAtomic)
{
  using Reduction = legate::MinReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  // EXCLUSIVE = false triggers AtomicWrapper<Minimum>::operator()<__half>
  Reduction::fold<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 3.0F);

  // Test the other direction
  lhs = legate::Half{K_FLOAT_2_0};
  const legate::Half rhs2{7.0F};
  Reduction::fold<false>(lhs, rhs2);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 2.0F);
}

TEST_F(RedopUnit, HalfMaxReductionApplyAtomic)
{
  using Reduction = legate::MaxReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{8.0F};

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);
}

TEST_F(RedopUnit, HalfMinReductionApplyAtomic)
{
  using Reduction = legate::MinReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 3.0F);
}

TEST_F(RedopUnit, HalfSumReductionFoldAtomic)
{
  using Reduction = legate::SumReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::fold<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 8.0F);
}

TEST_F(RedopUnit, HalfProdReductionFoldAtomic)
{
  using Reduction = legate::ProdReduction<legate::Half>;

  legate::Half lhs{K_FLOAT_5_0};
  const legate::Half rhs{3.0F};

  Reduction::fold<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs), 15.0F);
}

}  // namespace

}  // namespace redop_test
