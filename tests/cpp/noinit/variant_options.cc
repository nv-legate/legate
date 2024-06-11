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

namespace variant_options_test {

using VariantOptions = DefaultFixture;

TEST_F(VariantOptions, Basic)
{
  legate::VariantOptions options;
  EXPECT_EQ(options.leaf, true);
  EXPECT_EQ(options.inner, false);
  EXPECT_EQ(options.idempotent, false);
  EXPECT_EQ(options.concurrent, false);
  EXPECT_EQ(options.return_size, legate::LEGATE_MAX_SIZE_SCALAR_RETURN);

  options.with_leaf(false);
  EXPECT_EQ(options.leaf, false);

  options.with_inner(true);
  EXPECT_EQ(options.inner, true);

  options.with_idempotent(true);
  EXPECT_EQ(options.idempotent, true);

  options.with_concurrent(true);
  EXPECT_EQ(options.concurrent, true);

  options.with_return_size(256);  // NOLINT(readability-magic-numbers)
  EXPECT_EQ(options.return_size, 256);
}

}  // namespace variant_options_test
