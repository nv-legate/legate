/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace variant_options_test {

using VariantOptions = DefaultFixture;

TEST_F(VariantOptions, Basic)
{
  legate::VariantOptions options;

  ASSERT_EQ(options.concurrent, false);
  ASSERT_EQ(options, legate::VariantOptions::DEFAULT_OPTIONS);

  options.with_concurrent(true);
  ASSERT_EQ(options.concurrent, true);
}

TEST_F(VariantOptions, Options)
{
  const auto options = legate::VariantOptions{}
                         .with_concurrent(true)
                         .with_has_allocations(true)
                         .with_elide_device_ctx_sync(true)
                         .with_has_side_effect(true)
                         .with_may_throw_exception(true)
                         .with_communicators({"my_comm", "my_other_comm"});

  ASSERT_EQ(options.concurrent, true);
  ASSERT_EQ(options.has_allocations, true);
  ASSERT_EQ(options.elide_device_ctx_sync, true);
  ASSERT_EQ(options.has_side_effect, true);
  ASSERT_EQ(options.may_throw_exception, true);
  ASSERT_THAT(
    options.communicators,
    ::testing::Optional(::testing::ElementsAreArray(
      {std::string_view{"my_comm"}, std::string_view{"my_other_comm"}, std::string_view{}})));
}

TEST_F(VariantOptions, CommConcurrent)
{
  auto options = legate::VariantOptions{}.with_concurrent(false);

  ASSERT_EQ(options.concurrent, false);
  ASSERT_EQ(options.communicators, std::nullopt);

  options.with_communicators({"my_comm"});

  ASSERT_EQ(options.concurrent, true);
  ASSERT_THAT(options.communicators,
              ::testing::Optional(::testing::ElementsAreArray(
                {std::string_view{"my_comm"}, std::string_view{}, std::string_view{}})));
}

}  // namespace variant_options_test
