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

namespace variant_options_test {

using VariantOptions = DefaultFixture;

TEST_F(VariantOptions, Basic)
{
  legate::VariantOptions options;
  EXPECT_EQ(options.concurrent, false);

  options.with_concurrent(true);
  EXPECT_EQ(options.concurrent, true);
}

}  // namespace variant_options_test
