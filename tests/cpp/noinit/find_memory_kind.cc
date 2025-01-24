/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

namespace test_find_memory_kind_noinit {

using FindMemoryKind = ::testing::Test;

TEST_F(FindMemoryKind, BeforeInit)
{
  EXPECT_THROW(static_cast<void>(legate::find_memory_kind_for_executing_processor()),
               std::runtime_error);
}

}  // namespace test_find_memory_kind_noinit
