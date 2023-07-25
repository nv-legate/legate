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
#include "task_hello.h"

namespace hello {

legate::LogicalStore iota(legate::Library library, size_t size)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::hello::IOTA);
  auto output  = runtime->create_store({size}, legate::float32(), true);
  auto part    = task.declare_partition();
  task.add_output(output, part);
  runtime->submit(std::move(task));
  return output;
}

legate::LogicalStore square(legate::Library library, legate::LogicalStore input)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::hello::SQUARE);
  auto output  = runtime->create_store(input.extents().data(), legate::float32(), true);
  auto part1   = task.declare_partition();
  auto part2   = task.declare_partition();

  task.add_input(input, part1);
  task.add_output(output, part2);
  runtime->submit(std::move(task));
  return output;
}

legate::LogicalStore sum(legate::Library library, legate::LogicalStore input, const void* bytearray)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::hello::SUM);

  auto output = runtime->create_store(legate::Scalar(legate::float32(), bytearray));

  auto part1 = task.declare_partition();
  auto part2 = task.declare_partition();

  task.add_input(input, part1);
  task.add_reduction(output, legate::ReductionOpKind::ADD, part2);
  runtime->submit(std::move(task));
  return output;
}

float to_scalar(legate::Library library, legate::LogicalStore scalar)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto p_scalar = scalar.get_physical_store();
  auto acc      = p_scalar.read_accessor<float, 1>();
  float output  = static_cast<float>(acc[{0}]);
  return output;
}

TEST(Example, Hello)
{
  legate::Core::perform_registration<task::hello::register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::hello::library_name);

  float bytearray = 0;

  auto store        = iota(library, 5);
  auto storeSquared = square(library, store);
  auto storeSummed  = sum(library, storeSquared, &bytearray);
  float scalar      = to_scalar(library, storeSummed);

  ASSERT_EQ(scalar, 55);
}

}  // namespace hello
