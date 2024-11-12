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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace string_type_test {

namespace {

using StringTypeUnit = DefaultFixture;

}  // namespace

TEST_F(StringTypeUnit, StringType)
{
  const legate::Type type = legate::string_type();

  ASSERT_EQ(type.code(), legate::Type::Code::STRING);
  ASSERT_THROW(static_cast<void>(type.size()), std::invalid_argument);
  ASSERT_EQ(type.alignment(), alignof(std::max_align_t));
  ASSERT_TRUE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), "string");

  // Note: aim to test the copy initialization of Type
  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);
}

}  // namespace string_type_test
