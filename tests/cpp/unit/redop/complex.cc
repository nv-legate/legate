/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <unit/redop/common.h>

namespace redop_test {

namespace {

// ==========================================================================================
// Test Complex<float> reductions - Identity
// ==========================================================================================

TEST_F(RedopUnit, ComplexFloatSumReductionIdentity)
{
  using Reduction = legate::SumReduction<legate::Complex<float>>;

  ASSERT_FLOAT_EQ(Reduction::identity.real(), 0.0F);
  ASSERT_FLOAT_EQ(Reduction::identity.imag(), 0.0F);
}

TEST_F(RedopUnit, ComplexFloatProdReductionIdentity)
{
  using Reduction = legate::ProdReduction<legate::Complex<float>>;

  ASSERT_FLOAT_EQ(Reduction::identity.real(), 1.0F);
  ASSERT_FLOAT_EQ(Reduction::identity.imag(), 0.0F);
}

// ==========================================================================================
// Test Complex<float> reductions - Fold
// ==========================================================================================

TEST_F(RedopUnit, ComplexFloatSumReductionFold)
{
  using Reduction = legate::SumReduction<legate::Complex<float>>;

  legate::Complex<float> lhs{K_FLOAT_3_0, K_FLOAT_4_0};
  const legate::Complex<float> rhs{1.0F, 2.0F};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs.real(), 4.0F);
  ASSERT_FLOAT_EQ(lhs.imag(), 6.0F);
}

TEST_F(RedopUnit, ComplexFloatProdReductionFold)
{
  using Reduction = legate::ProdReduction<legate::Complex<float>>;

  // (3 + 4i) * (1 + 2i) = 3 + 6i + 4i + 8i^2 = 3 + 10i - 8 = -5 + 10i
  legate::Complex<float> lhs{K_FLOAT_3_0, K_FLOAT_4_0};
  const legate::Complex<float> rhs{1.0F, 2.0F};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs.real(), -5.0F);
  ASSERT_FLOAT_EQ(lhs.imag(), 10.0F);
}

// ==========================================================================================
// Test Complex<double> reductions - Identity
// ==========================================================================================

TEST_F(RedopUnit, ComplexDoubleSumReductionIdentity)
{
  using Reduction = legate::SumReduction<legate::Complex<double>>;

  ASSERT_DOUBLE_EQ(Reduction::identity.real(), 0.0);
  ASSERT_DOUBLE_EQ(Reduction::identity.imag(), 0.0);
}

// ==========================================================================================
// Test Complex<double> reductions - Fold
// ==========================================================================================

TEST_F(RedopUnit, ComplexDoubleSumReductionFold)
{
  using Reduction = legate::SumReduction<legate::Complex<double>>;

  legate::Complex<double> lhs{K_DOUBLE_3_0, K_DOUBLE_4_0};
  const legate::Complex<double> rhs{1.0, 2.0};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_DOUBLE_EQ(lhs.real(), 4.0);
  ASSERT_DOUBLE_EQ(lhs.imag(), 6.0);
}

// ==========================================================================================
// Test Complex<Half> reductions - Identity
// ==========================================================================================

TEST_F(RedopUnit, ComplexHalfSumReductionIdentity)
{
  using Reduction = legate::SumReduction<legate::Complex<legate::Half>>;

  ASSERT_FLOAT_EQ(static_cast<float>(Reduction::identity.real()), 0.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(Reduction::identity.imag()), 0.0F);
}

TEST_F(RedopUnit, ComplexHalfProdReductionIdentity)
{
  using Reduction = legate::ProdReduction<legate::Complex<legate::Half>>;

  ASSERT_FLOAT_EQ(static_cast<float>(Reduction::identity.real()), 1.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(Reduction::identity.imag()), 0.0F);
}

// ==========================================================================================
// Test Complex<Half> reductions - Fold
// ==========================================================================================

TEST_F(RedopUnit, ComplexHalfSumReductionFold)
{
  using Reduction = legate::SumReduction<legate::Complex<legate::Half>>;

  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), 4.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 6.0F);
}

TEST_F(RedopUnit, ComplexHalfProdReductionFold)
{
  using Reduction = legate::ProdReduction<legate::Complex<legate::Half>>;

  // (3 + 4i) * (1 + 2i) = -5 + 10i
  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::fold<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), -5.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 10.0F);
}

// ==========================================================================================
// Test Complex reductions - Apply
// ==========================================================================================

TEST_F(RedopUnit, ComplexFloatSumReductionApply)
{
  using Reduction = legate::SumReduction<legate::Complex<float>>;

  legate::Complex<float> lhs{K_FLOAT_3_0, K_FLOAT_4_0};
  const legate::Complex<float> rhs{1.0F, 2.0F};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(lhs.real(), 4.0F);
  ASSERT_FLOAT_EQ(lhs.imag(), 6.0F);
}

TEST_F(RedopUnit, ComplexDoubleSumReductionApply)
{
  using Reduction = legate::SumReduction<legate::Complex<double>>;

  legate::Complex<double> lhs{K_DOUBLE_3_0, K_DOUBLE_4_0};
  const legate::Complex<double> rhs{1.0, 2.0};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_DOUBLE_EQ(lhs.real(), 4.0);
  ASSERT_DOUBLE_EQ(lhs.imag(), 6.0);
}

// ==========================================================================================
// Test Complex<Half> reductions - Apply (EXCLUSIVE = true)
// These cover BaseReduction<complex<__half>, *, AtomicWrapperComplexHalf>::apply<true>
// ==========================================================================================

TEST_F(RedopUnit, ComplexHalfSumReductionApply)
{
  using Reduction = legate::SumReduction<legate::Complex<legate::Half>>;

  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), 4.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 6.0F);
}

TEST_F(RedopUnit, ComplexHalfProdReductionApply)
{
  using Reduction = legate::ProdReduction<legate::Complex<legate::Half>>;

  // (3 + 4i) * (1 + 2i) = -5 + 10i
  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::apply<true>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), -5.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 10.0F);
}

// ==========================================================================================
// Test Complex<Half> reductions - Atomic (EXCLUSIVE = false)
// These cover BaseReduction<complex<__half>, *, AtomicWrapperComplexHalf>::apply<false>
// ==========================================================================================

TEST_F(RedopUnit, ComplexHalfSumReductionFoldAtomic)
{
  using Reduction = legate::SumReduction<legate::Complex<legate::Half>>;

  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::fold<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), 4.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 6.0F);
}

TEST_F(RedopUnit, ComplexHalfSumReductionApplyAtomic)
{
  using Reduction = legate::SumReduction<legate::Complex<legate::Half>>;

  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), 4.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 6.0F);
}

TEST_F(RedopUnit, ComplexHalfProdReductionFoldAtomic)
{
  using Reduction = legate::ProdReduction<legate::Complex<legate::Half>>;

  // (3 + 4i) * (1 + 2i) = -5 + 10i
  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::fold<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), -5.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 10.0F);
}

TEST_F(RedopUnit, ComplexHalfProdReductionApplyAtomic)
{
  using Reduction = legate::ProdReduction<legate::Complex<legate::Half>>;

  // (3 + 4i) * (1 + 2i) = -5 + 10i
  legate::Complex<legate::Half> lhs{legate::Half{K_FLOAT_3_0}, legate::Half{K_FLOAT_4_0}};
  const legate::Complex<legate::Half> rhs{legate::Half{1.0F}, legate::Half{2.0F}};

  Reduction::apply<false>(lhs, rhs);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.real()), -5.0F);
  ASSERT_FLOAT_EQ(static_cast<float>(lhs.imag()), 10.0F);
}

}  // namespace

}  // namespace redop_test
