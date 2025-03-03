/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_signature.h>
#include <legate/task/task_signature.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace nargs_util {

inline void test_single_value(const legate::detail::TaskSignature::Nargs& nargs,
                              std::uint32_t value)
{
  ASSERT_THAT(nargs.value(), ::testing::VariantWith<std::uint32_t>(value));
  ASSERT_EQ(nargs.upper_limit(), value);
  ASSERT_TRUE(nargs.compatible_with(value));
  ASSERT_FALSE(nargs.compatible_with(value + 1));
  ASSERT_FALSE(nargs.compatible_with(value - 1));
}

inline void test_bounded_range(const legate::detail::TaskSignature::Nargs& nargs,
                               std::uint32_t low_bound,
                               std::uint32_t hi_bound)
{
  using pair_type = std::pair<std::uint32_t, std::uint32_t>;

  ASSERT_THAT(nargs.value(),
              ::testing::VariantWith<pair_type>(::testing::Pair(low_bound, hi_bound)));
  ASSERT_EQ(nargs.upper_limit(), hi_bound);
  ASSERT_TRUE(nargs.compatible_with(low_bound));
  ASSERT_TRUE(nargs.compatible_with(hi_bound));
  ASSERT_FALSE(nargs.compatible_with(hi_bound + 1));
  ASSERT_FALSE(nargs.compatible_with(low_bound - 1));
}

inline void test_unbounded_range(const legate::detail::TaskSignature::Nargs& nargs,
                                 std::uint32_t low_bound)
{
  using pair_type = std::pair<std::uint32_t, std::uint32_t>;

  ASSERT_THAT(nargs.value(),
              ::testing::VariantWith<pair_type>(
                ::testing::Pair(low_bound, legate::TaskSignature::UNBOUNDED)));
  ASSERT_EQ(nargs.upper_limit(), low_bound);
  ASSERT_TRUE(nargs.compatible_with(low_bound));
  ASSERT_TRUE(nargs.compatible_with(low_bound + 1'000));
  ASSERT_FALSE(nargs.compatible_with(low_bound - 1));
}

}  // namespace nargs_util
