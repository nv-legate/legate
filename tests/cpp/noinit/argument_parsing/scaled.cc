/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/argument.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace test_scaled {

using ArgumentTypes = ::testing::Types<std::uint32_t, std::int64_t>;

template <typename T>
class ScaledUnit : public DefaultFixture {};

TYPED_TEST_SUITE(ScaledUnit, ArgumentTypes, );

TYPED_TEST(ScaledUnit, DefaultConstructNoScale)
{
  constexpr auto VALUE = 123;
  constexpr auto SCALE = 1;
  auto scal            = legate::detail::Scaled<TypeParam>{VALUE, SCALE};

  ASSERT_EQ(scal.unscaled_value(), VALUE);
  ASSERT_EQ(scal.unscaled_value_mut(), VALUE);
  ASSERT_EQ(scal.scaled_value(), VALUE * SCALE);
  ASSERT_EQ(scal.scale(), SCALE);
}

TYPED_TEST(ScaledUnit, DefaultConstructScale)
{
  constexpr auto VALUE = 123;
  constexpr auto SCALE = 10;
  auto scal            = legate::detail::Scaled<TypeParam>{VALUE, SCALE};

  ASSERT_EQ(scal.unscaled_value(), VALUE);
  ASSERT_EQ(scal.unscaled_value_mut(), VALUE);
  ASSERT_EQ(scal.scaled_value(), VALUE * SCALE);
  ASSERT_EQ(scal.scale(), SCALE);
}

TYPED_TEST(ScaledUnit, Mutate)
{
  constexpr auto DEFAULT_VALUE = 123;
  constexpr auto SCALE         = 10;
  auto scal                    = legate::detail::Scaled<TypeParam>{DEFAULT_VALUE, SCALE};

  constexpr auto NEW_VALUE = 345;

  scal.unscaled_value_mut() = NEW_VALUE;

  ASSERT_EQ(scal.unscaled_value(), NEW_VALUE);
  ASSERT_EQ(scal.unscaled_value_mut(), NEW_VALUE);
  ASSERT_EQ(scal.scaled_value(), NEW_VALUE * SCALE);
  ASSERT_EQ(scal.scale(), SCALE);
}

TYPED_TEST(ScaledUnit, ToString)
{
  constexpr auto VALUE = 123;
  constexpr auto SCALE = 10;
  const auto scal      = legate::detail::Scaled<TypeParam>{VALUE, SCALE};

  std::stringstream ss;
  std::stringstream ss_expected;

  ss << scal;
  ss_expected << "Scaled(scale: " << SCALE << ", value: " << VALUE << ")";

  ASSERT_EQ(ss.str(), ss_expected.str());
}

}  // namespace test_scaled
