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

#include "legate_defines.h"
#include "utilities/sanitizer_options.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  GTEST_FLAG_SET(death_test_style, "threadsafe");

  return RUN_ALL_TESTS();
}
