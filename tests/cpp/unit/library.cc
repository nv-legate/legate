/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "legate.h"

TEST(Library, Create)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library("libA");
  EXPECT_EQ(lib, runtime->find_library("libA"));
  EXPECT_EQ(lib, runtime->maybe_find_library("libA").value());
}

TEST(Library, FindOrCreate)
{
  auto* runtime = legate::Runtime::get_runtime();

  legate::ResourceConfig config;
  config.max_tasks = 1;

  bool created = false;
  auto p_lib1  = runtime->find_or_create_library("libA", config, nullptr, &created);
  EXPECT_TRUE(created);

  config.max_tasks = 2;
  auto p_lib2      = runtime->find_or_create_library("libA", config, nullptr, &created);
  EXPECT_FALSE(created);
  EXPECT_EQ(p_lib1, p_lib2);
  EXPECT_TRUE(p_lib2.valid_task_id(p_lib2.get_task_id(0)));
  EXPECT_FALSE(p_lib2.valid_task_id(p_lib2.get_task_id(1)));
}

TEST(Library, FindNonExistent)
{
  auto* runtime = legate::Runtime::get_runtime();

  EXPECT_THROW(runtime->find_library("libB"), std::out_of_range);

  EXPECT_EQ(runtime->maybe_find_library("libB"), std::nullopt);
}
