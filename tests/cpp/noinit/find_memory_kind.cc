/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

namespace test_find_memory_kind_noinit {

using FindMemoryKind = ::testing::Test;

TEST_F(FindMemoryKind, BeforeInit)
{
  EXPECT_THROW(static_cast<void>(legate::find_memory_kind_for_executing_processor()),
               std::runtime_error);
}

}  // namespace test_find_memory_kind_noinit
