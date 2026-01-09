/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task_signature.h>
#include <legate/task/task_signature.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

#include <unit/task_signature/nargs_util.h>
#include <utilities/utilities.h>

namespace test_task_signature_nargs {

class TaskSignatureNargsUnit : public DefaultFixture {};

TEST_F(TaskSignatureNargsUnit, Basic)
{
  constexpr auto nargs = legate::detail::TaskSignature::Nargs{};

  nargs_util::test_single_value(nargs, 0);
}

TEST_F(TaskSignatureNargsUnit, SingleValue)
{
  constexpr auto VALUE = std::uint32_t{10};
  const auto nargs     = legate::detail::TaskSignature::Nargs{VALUE};

  nargs_util::test_single_value(nargs, VALUE);
}

TEST_F(TaskSignatureNargsUnit, BoundedRange)
{
  constexpr auto LOW_BOUND = std::uint32_t{10};
  constexpr auto HI_BOUND  = std::uint32_t{11};
  const auto nargs         = legate::detail::TaskSignature::Nargs{LOW_BOUND, HI_BOUND};

  nargs_util::test_bounded_range(nargs, LOW_BOUND, HI_BOUND);
}

TEST_F(TaskSignatureNargsUnit, UnboundedRange)
{
  constexpr auto LOW_BOUND = std::uint32_t{10};
  const auto nargs =
    legate::detail::TaskSignature::Nargs{LOW_BOUND, legate::TaskSignature::UNBOUNDED};

  nargs_util::test_unbounded_range(nargs, LOW_BOUND);
}

class Invalid : public TaskSignatureNargsUnit,
                public ::testing::WithParamInterface<std::pair<std::uint32_t, std::uint32_t>> {};

INSTANTIATE_TEST_SUITE_P(TaskSignatureNargsUnit,
                         Invalid,
                         ::testing::Values(std::make_pair(0, 0),
                                           std::make_pair(legate::TaskSignature::UNBOUNDED, 10),
                                           std::make_pair(2, 1)));

TEST_P(Invalid, InvalidRange)
{
  // Captured structured bindings are a C++20 extension.
  LEGATE_CPP_VERSION_TODO(20, "Use structured bindigs below");
  const auto lo     = GetParam().first;
  const auto hi     = GetParam().second;
  const auto lambda = [&] { std::ignore = legate::detail::TaskSignature::Nargs{lo, hi}; };

  ASSERT_THAT(
    lambda,
    ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr(fmt::format(
      "Invalid argument range: {}, upper bound must be strictly greater than the lower bound",
      GetParam()))));
}

}  // namespace test_task_signature_nargs
