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
#include "tasks/task_simple.h"

namespace inout {

void test_inout()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::simple::library_name);

  auto store = runtime->create_store({10, 10}, legate::int64());
  runtime->issue_fill(store, legate::Scalar(int64_t(0)));

  auto task = runtime->create_task(context, task::simple::HELLO);
  task->add_input(store, task->find_or_declare_partition(store));
  task->add_output(store, task->find_or_declare_partition(store));
  runtime->submit(std::move(task));
}

TEST(Integration, InOut)
{
  legate::Core::perform_registration<task::simple::register_tasks>();
  test_inout();
}

}  // namespace inout
