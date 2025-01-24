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

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_shutdown_callback {

using ShutdownCallback = DefaultFixture;

TEST_F(ShutdownCallback, Basic)
{
  bool callback_called_1 = false;
  auto callback_1        = [&callback_called_1]() noexcept { callback_called_1 = true; };

  bool callback_called_2 = false;
  auto callback_2        = [&callback_called_2]() noexcept { callback_called_2 = true; };

  legate::register_shutdown_callback(std::move(callback_1));
  legate::register_shutdown_callback(std::move(callback_2));

  // Shutdown and verify that the callback is called
  ASSERT_EQ(legate::finish(), 0);

  ASSERT_TRUE(callback_called_1);
  ASSERT_TRUE(callback_called_2);
}

TEST_F(ShutdownCallback, Nested)
{
  bool callback_called = false;
  auto callback_inside = [&callback_called]() noexcept { callback_called = true; };
  auto callback        = [&callback_inside]() noexcept {
    legate::register_shutdown_callback(std::move(callback_inside));
  };

  legate::register_shutdown_callback(std::move(callback));

  // Shutdown and verify that the callback is called
  ASSERT_EQ(legate::finish(), 0);

  ASSERT_TRUE(callback_called);
}

}  // namespace test_shutdown_callback
