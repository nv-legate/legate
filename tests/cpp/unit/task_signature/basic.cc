/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/proxy/align.h>
#include <legate/partitioning/detail/proxy/select.h>
#include <legate/partitioning/proxy.h>
#include <legate/task/detail/task_signature.h>
#include <legate/task/task_signature.h>

#include <fmt/format.h>

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

TEST_F(TaskSignatureUnit, Constraints)
{
  const auto signature =
    legate::TaskSignature{}.constraints({{legate::align(legate::proxy::inputs)}});
  const auto& sig = signature.impl();

  ASSERT_FALSE(sig->inputs().has_value());
  ASSERT_FALSE(sig->outputs().has_value());
  ASSERT_FALSE(sig->scalars().has_value());
  ASSERT_FALSE(sig->redops().has_value());
  ASSERT_TRUE(sig->constraints().has_value());
}

TEST_F(TaskSignatureUnit, InvalidConstraint)
{
  const auto signature = legate::TaskSignature{}.inputs(1).outputs(1).constraints(
    {{legate::align(legate::proxy::inputs[-1], legate::proxy::outputs[0])}});
  const auto& sig = signature.impl();

  constexpr std::string_view task_name = "InvalidConstraint";

  ASSERT_THAT([&] { sig->validate(task_name); },
              ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr(
                fmt::format("Invalid task signature for task {}", task_name))));
}

using TaskSignatureDeathTest = TaskSignatureUnit;

TEST_F(TaskSignatureDeathTest, InvalidProxyArray)
{
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)
  const auto invalid_kind =
    static_cast<legate::ProxyArrayArgument::Kind>(99);  // Invalid array argument kind
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)
  const legate::ProxyArrayArgument proxy_array_argument{invalid_kind, 0};
  const legate::detail::ArgSelectVisitor visitor;

  ASSERT_EXIT(static_cast<void>(visitor(proxy_array_argument)),
              ::testing::KilledBySignal{SIGABRT},
              "Unhandled array kind");
}

namespace {

// NOLINTBEGIN
MATCHER(PtrEq, "Pointer equals") { return *std::get<0>(arg) == *std::get<1>(arg); }

MATCHER(PtrNotEq, "Pointer not equals") { return *std::get<0>(arg) != *std::get<1>(arg); }

// NOLINTEND

}  // namespace

class SameConstraint : public DefaultFixture,
                       public ::testing::WithParamInterface<
                         std::tuple<legate::ProxyConstraint, legate::ProxyConstraint>> {};

INSTANTIATE_TEST_SUITE_P(
  TaskSignatureUnit,
  SameConstraint,
  ::testing::Values(
    std::make_tuple(legate::align(legate::proxy::inputs),
                    legate::align(legate::proxy::inputs[0], legate::proxy::inputs)),
    std::make_tuple(legate::align(legate::proxy::outputs),
                    legate::align(legate::proxy::outputs[0], legate::proxy::outputs)),
    std::make_tuple(legate::broadcast(legate::proxy::inputs),
                    legate::broadcast(legate::proxy::inputs)),
    std::make_tuple(legate::bloat(legate::proxy::inputs, legate::proxy::outputs, {0}, {1}),
                    legate::bloat(legate::proxy::inputs, legate::proxy::outputs, {0}, {1})),
    std::make_tuple(legate::image(legate::proxy::inputs, legate::proxy::outputs),
                    legate::image(legate::proxy::inputs, legate::proxy::outputs)),
    std::make_tuple(legate::scale({2}, legate::proxy::inputs, legate::proxy::outputs),
                    legate::scale({2}, legate::proxy::inputs, legate::proxy::outputs))));

class DifferentConstraint : public DefaultFixture,
                            public ::testing::WithParamInterface<
                              std::tuple<legate::ProxyConstraint, legate::ProxyConstraint>> {};

INSTANTIATE_TEST_SUITE_P(
  TaskSignatureUnit,
  DifferentConstraint,
  ::testing::Values(
    std::make_tuple(legate::align(legate::proxy::inputs), legate::align(legate::proxy::outputs)),
    std::make_tuple(legate::broadcast(legate::proxy::inputs),
                    legate::broadcast(legate::proxy::inputs, legate::tuple<std::uint32_t>{0})),
    std::make_tuple(legate::bloat(legate::proxy::inputs, legate::proxy::outputs, {0}, {1}),
                    legate::bloat(legate::proxy::inputs, legate::proxy::outputs, {0}, {0})),
    std::make_tuple(legate::image(legate::proxy::inputs, legate::proxy::outputs),
                    legate::image(legate::proxy::inputs,
                                  legate::proxy::outputs,
                                  legate::ImageComputationHint::NO_HINT)),
    std::make_tuple(legate::scale({2}, legate::proxy::inputs, legate::proxy::outputs),
                    legate::scale({2}, legate::proxy::inputs, legate::proxy::inputs)),
    std::make_tuple(legate::align(legate::proxy::inputs),
                    legate::broadcast(legate::proxy::inputs, legate::tuple<std::uint32_t>{0})),
    std::make_tuple(legate::broadcast(legate::proxy::inputs), legate::align(legate::proxy::inputs)),
    std::make_tuple(legate::bloat(legate::proxy::inputs, legate::proxy::outputs, {0}, {1}),
                    legate::align(legate::proxy::inputs)),
    std::make_tuple(legate::image(legate::proxy::inputs, legate::proxy::outputs),
                    legate::broadcast(legate::proxy::inputs)),
    std::make_tuple(legate::scale({2}, legate::proxy::inputs, legate::proxy::outputs),
                    legate::image(legate::proxy::inputs, legate::proxy::outputs))));

TEST_P(SameConstraint, Check)
{
  const auto [cstrnt1, cstrnt2] = GetParam();
  const auto signature          = legate::TaskSignature{}.constraints({{cstrnt1}});
  const auto& sig               = signature.impl();
  const auto& sig_constraints   = sig->constraints();

  ASSERT_TRUE(sig_constraints.has_value());

  const auto& cstrnts = *sig_constraints;  // NOLINT(bugprone-unchecked-optional-access)
  const auto check    = {cstrnt2.impl()};

  ASSERT_THAT(cstrnts, ::testing::Pointwise(PtrEq(), check));
}

TEST_P(DifferentConstraint, Check)
{
  const auto [cstrnt1, cstrnt2] = GetParam();
  const auto signature          = legate::TaskSignature{}.constraints({{cstrnt1}});
  const auto& sig               = signature.impl();
  const auto& sig_constraints   = sig->constraints();

  ASSERT_TRUE(sig_constraints.has_value());

  const auto& cstrnts = *sig_constraints;  // NOLINT(bugprone-unchecked-optional-access)
  const auto check    = {cstrnt2.impl()};

  ASSERT_THAT(cstrnts, ::testing::Pointwise(PtrNotEq(), check));
}

}  // namespace test_task_signature_basic
