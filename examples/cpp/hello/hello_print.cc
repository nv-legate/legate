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

#include "core/mapping/mapping.h"
#include "legate.h"
#include "task_hello.h"

namespace helloprint {

void test_print_hello(legate::LibraryContext* context, std::string str)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, task::hello::HELLO_WORLD);

  task->add_scalar_arg(legate::Scalar(str));
  runtime->submit(std::move(task));
}

void test_print_hellos(legate::LibraryContext* context, std::string str, size_t count)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, task::hello::HELLO_WORLD, legate::Shape({count}));

  task->add_scalar_arg(legate::Scalar(str));
  runtime->submit(std::move(task));
}

TEST(Example, HelloPrint)
{
  legate::Core::perform_registration<task::hello::register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::hello::library_name);

  test_print_hello(context, "Hello World!");
  test_print_hellos(context, "Hello World! x3", 3);
}

}  // namespace helloprint
