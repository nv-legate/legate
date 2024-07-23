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

#include "core/runtime/detail/runtime.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace runtime_exception_test {

using RuntimeNoInit = ::testing::Test;

TEST_F(RuntimeNoInit, GetRuntime)
{
  ASSERT_EQ(legate::start(0, nullptr), 0);
  ASSERT_EQ(legate::finish(), 0);
  EXPECT_THROW(static_cast<void>(legate::Runtime::get_runtime()), std::runtime_error);
}

}  // namespace runtime_exception_test
