/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/runtime/detail/config.h>

#include <legate.h>

#include <legate/utilities/detail/env.h>

#include <gtest/gtest.h>

#include <utilities/env.h>

namespace config_test {

using ConfigTest = ::testing::Test;

namespace {

void validate_config()
{
  ASSERT_FALSE(legate::detail::Config::get_config().parsed());
  ASSERT_TRUE(legate::detail::Config::get_config().auto_config());
  ASSERT_FALSE(legate::detail::Config::get_config().show_config());
  ASSERT_FALSE(legate::detail::Config::get_config().show_progress_requested());
  ASSERT_FALSE(legate::detail::Config::get_config().use_empty_task());
  ASSERT_FALSE(legate::detail::Config::get_config().synchronize_stream_view());
  ASSERT_FALSE(legate::detail::Config::get_config().log_mapping_decisions());
  ASSERT_FALSE(legate::detail::Config::get_config().log_partitioning_decisions());
  ASSERT_FALSE(legate::detail::Config::get_config().has_socket_mem());
  ASSERT_FALSE(legate::detail::Config::get_config().warmup_nccl());
  ASSERT_FALSE(legate::detail::Config::get_config().enable_inline_task_launch());
  ASSERT_EQ(legate::detail::Config::get_config().num_omp_threads(), 0);
}

}  // namespace

TEST_F(ConfigTest, Reset)
{
  // Validate config before testing
  validate_config();

  // Set some config before testing
  legate::detail::Config::get_config_mut().set_show_config(true);
  legate::detail::Config::get_config_mut().set_num_omp_threads(1);

  ASSERT_TRUE(legate::detail::Config::get_config().show_config());
  ASSERT_EQ(legate::detail::Config::get_config().num_omp_threads(), 1);

  // Set invalid environment parameter
  const auto _ = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_AUTO_CONFIG, "INVALID_VAL", true);

  // Should trigger exception and call reset_ function
  ASSERT_THROW(legate::detail::Config::get_config_mut().parse(), std::invalid_argument);

  // Check whether the configs are restored after reset
  validate_config();
}

}  // namespace config_test
