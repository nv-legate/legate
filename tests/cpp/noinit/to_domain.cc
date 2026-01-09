/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/tuple.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace to_domain_test {

using ToDomain = DefaultFixture;

TEST_F(ToDomain, Empty)
{
  const auto shape    = legate::tuple<std::uint64_t>{};
  const auto domain   = legate::detail::to_domain(shape);
  const auto expected = legate::Domain{0, 0};

  ASSERT_EQ(expected, domain);
}

TEST_F(ToDomain, 1D)
{
  const auto shape    = legate::tuple<std::uint64_t>{1};
  const auto domain   = legate::detail::to_domain(shape);
  const auto expected = legate::Domain{0, 0};

  ASSERT_EQ(expected, domain);
}

}  // namespace to_domain_test
