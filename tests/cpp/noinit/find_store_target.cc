/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

namespace test_find_store_target_noinit {

using FindStoreTarget = ::testing::Test;

TEST_F(FindStoreTarget, BeforeInit)
{
  EXPECT_THROW(static_cast<void>(legate::find_store_target_for_executing_processor()),
               std::runtime_error);
}

}  // namespace test_find_store_target_noinit
