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

#include "legate.h"
#include "task_hello.h"

#include <gtest/gtest.h>

namespace helloprint {

void test_print_hello(legate::Library library, std::string str)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::hello::HELLO_WORLD);

  task.add_scalar_arg(legate::Scalar(str));
  runtime->submit(std::move(task));
}

void test_print_hellos(legate::Library library, std::string str, std::size_t count)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::hello::HELLO_WORLD, legate::Shape({count}));

  task.add_scalar_arg(legate::Scalar(str));
  runtime->submit(std::move(task));
}

TEST(Example, HelloPrint)
{
  task::hello::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::hello::library_name);

  test_print_hello(library, "Hello World!");
  test_print_hellos(library, "Hello World! x3", 3);
}

}  // namespace helloprint
