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

#include <gtest/gtest.h>

#include "legate.h"

namespace test_library {

TEST(Library, Create)
{
  const char* LIBNAME = "test_library.libA";
  auto* runtime       = legate::Runtime::get_runtime();
  auto lib            = runtime->create_library(LIBNAME);
  EXPECT_EQ(lib, runtime->find_library(LIBNAME));
  EXPECT_EQ(lib, runtime->maybe_find_library(LIBNAME).value());
}

TEST(Library, FindOrCreate)
{
  const char* LIBNAME = "test_library.libB";

  auto* runtime = legate::Runtime::get_runtime();

  legate::ResourceConfig config;
  config.max_tasks = 1;

  bool created = false;
  auto p_lib1  = runtime->find_or_create_library(LIBNAME, config, nullptr, &created);
  EXPECT_TRUE(created);

  config.max_tasks = 2;
  auto p_lib2      = runtime->find_or_create_library(LIBNAME, config, nullptr, &created);
  EXPECT_FALSE(created);
  EXPECT_EQ(p_lib1, p_lib2);
  EXPECT_TRUE(p_lib2.valid_task_id(p_lib2.get_task_id(0)));
  EXPECT_FALSE(p_lib2.valid_task_id(p_lib2.get_task_id(1)));
}

TEST(Library, FindNonExistent)
{
  const char* LIBNAME = "test_library.libC";

  auto* runtime = legate::Runtime::get_runtime();

  EXPECT_THROW(runtime->find_library(LIBNAME), std::out_of_range);

  EXPECT_EQ(runtime->maybe_find_library(LIBNAME), std::nullopt);
}

}  // namespace test_library
