/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/proxy/align.h>
#include <legate/partitioning/proxy.h>
#include <legate/task/detail/task_signature.h>
#include <legate/task/task_signature.h>

#include <gtest/gtest.h>

#include <unit/task_signature/nargs_util.h>
#include <utilities/utilities.h>

namespace test_task_signature_basic {

class TaskSignatureUnit : public DefaultFixture {};

TEST_F(TaskSignatureUnit, Empty)
{
  const auto signature = legate::TaskSignature{};
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

namespace {

constexpr auto A_NUMBER = 123;

}  // namespace

TEST_F(TaskSignatureUnit, SingleInput)
{
  const auto signature = legate::TaskSignature{}.inputs(A_NUMBER);
  const auto& sig      = signature.impl();

  ASSERT_TRUE(sig->inputs().has_value());
  nargs_util::test_single_value(*sig->inputs(),  // NOLINT(bugprone-unchecked-optional-access)
                                A_NUMBER);
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, InputRangeBounded)
{
  constexpr auto low_bound = A_NUMBER;
  constexpr auto hi_bound  = A_NUMBER + 1;

  const auto signature = legate::TaskSignature{}.inputs(low_bound, hi_bound);
  const auto& sig      = signature.impl();

  ASSERT_TRUE(sig->inputs().has_value());
  nargs_util::test_bounded_range(*sig->inputs(),  // NOLINT(bugprone-unchecked-optional-access)
                                 low_bound,
                                 hi_bound);
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, SingleOutput)
{
  const auto signature = legate::TaskSignature{}.outputs(A_NUMBER);
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_TRUE(sig->outputs().has_value());
  nargs_util::test_single_value(*sig->outputs(),  // NOLINT(bugprone-unchecked-optional-access)
                                A_NUMBER);
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, OutputRangeBounded)
{
  constexpr auto low_bound = A_NUMBER;
  constexpr auto hi_bound  = A_NUMBER + 1;

  const auto signature = legate::TaskSignature{}.outputs(low_bound, hi_bound);
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_TRUE(sig->outputs().has_value());
  nargs_util::test_bounded_range(*sig->outputs(),  // NOLINT(bugprone-unchecked-optional-access)
                                 low_bound,
                                 hi_bound);
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, SingleScalar)
{
  const auto signature = legate::TaskSignature{}.scalars(A_NUMBER);
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_TRUE(sig->scalars().has_value());
  nargs_util::test_single_value(*sig->scalars(),  // NOLINT(bugprone-unchecked-optional-access)
                                A_NUMBER);
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, ScalarRangeBounded)
{
  constexpr auto low_bound = A_NUMBER;
  constexpr auto hi_bound  = A_NUMBER + 1;

  const auto signature = legate::TaskSignature{}.scalars(low_bound, hi_bound);
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_TRUE(sig->scalars().has_value());
  nargs_util::test_bounded_range(*sig->scalars(),  // NOLINT(bugprone-unchecked-optional-access)
                                 low_bound,
                                 hi_bound);
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, SingleRedop)
{
  const auto signature = legate::TaskSignature{}.redops(A_NUMBER);
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_TRUE(sig->redops().has_value());
  nargs_util::test_single_value(*sig->redops(),  // NOLINT(bugprone-unchecked-optional-access)
                                A_NUMBER);
  ASSERT_FALSE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, RedopRangeBounded)
{
  constexpr auto low_bound = A_NUMBER;
  constexpr auto hi_bound  = A_NUMBER + 1;

  const auto signature = legate::TaskSignature{}.redops(low_bound, hi_bound);
  const auto& sig      = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_TRUE(sig->redops().has_value());
  nargs_util::test_bounded_range(*sig->redops(),  // NOLINT(bugprone-unchecked-optional-access)
                                 low_bound,
                                 hi_bound);
  ASSERT_FALSE(sig->constraints().has_value());
}

namespace {

// NOLINTBEGIN
MATCHER(PtrEq, "Pointer equals") { return *std::get<0>(arg) == *std::get<1>(arg); }
// NOLINTEND

}  // namespace

TEST_F(TaskSignatureUnit, SomeConstraints)
{
  const auto signature =
    legate::TaskSignature{}.constraints({{legate::align(legate::proxy::inputs)}});
  const auto& sig = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());

  const auto& sig_constraints = sig->constraints();

  ASSERT_TRUE(sig_constraints.has_value());

  const auto& cstrnts = *sig_constraints;  // NOLINT(bugprone-unchecked-optional-access)

  const auto expected = {legate::align(legate::proxy::inputs[0], legate::proxy::inputs).impl()};

  ASSERT_THAT(cstrnts, ::testing::Pointwise(PtrEq(), expected));
}

}  // namespace test_task_signature_basic
