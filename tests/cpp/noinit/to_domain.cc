/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
