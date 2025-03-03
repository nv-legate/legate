/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate_defines.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  GTEST_FLAG_SET(death_test_style, "threadsafe");

  return RUN_ALL_TESTS();
}
