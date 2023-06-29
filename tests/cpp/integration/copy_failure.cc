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

namespace copy_failure {

void test_input_output_failure()
{
  // inconsistent number of inputs and outputs
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library("legate.core");

  std::vector<size_t> extents = {100};
  auto in_store1              = runtime->create_store(extents, legate::int64());
  auto in_store2              = runtime->create_store(extents, legate::int64());
  auto out_store              = runtime->create_store(extents, legate::int64());
  // fill input store with some values
  auto copy = runtime->create_copy();
  copy.add_input(in_store1);
  copy.add_input(in_store2);
  copy.add_output(out_store);
  EXPECT_THROW(runtime->submit(std::move(copy)), std::runtime_error);
}

void test_indirect_failure()
{
  // using indirect with several inputs/outputs
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library("legate.core");

  std::vector<size_t> extents = {100};
  auto in_store1              = runtime->create_store(extents, legate::int64());
  auto in_store2              = runtime->create_store(extents, legate::int64());
  auto out_store1             = runtime->create_store(extents, legate::int64());
  auto out_store2             = runtime->create_store(extents, legate::int64());
  auto indirect_store         = runtime->create_store(extents, legate::int64());

  auto copy = runtime->create_copy();
  copy.add_input(in_store1);
  copy.add_input(in_store2);
  copy.add_output(out_store1);
  copy.add_output(out_store2);
  copy.add_target_indirect(indirect_store);

  EXPECT_THROW(runtime->submit(std::move(copy)), std::runtime_error);
}

void test_shape_check_failure()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library("legate.core");

  auto store1 = runtime->create_store({10, 10}, legate::int64());
  auto store2 = runtime->create_store({5, 20}, legate::int64());
  auto store3 = runtime->create_store({20, 5}, legate::int64());

  {
    auto copy = runtime->create_copy();
    copy.add_input(store1);
    copy.add_output(store2);
    EXPECT_THROW(runtime->submit(std::move(copy)), std::runtime_error);
  }

  {
    auto copy = runtime->create_copy();
    copy.add_input(store1);
    copy.add_source_indirect(store2);
    copy.add_output(store3);
    EXPECT_THROW(runtime->submit(std::move(copy)), std::runtime_error);
  }

  {
    auto copy = runtime->create_copy();
    copy.add_input(store1);
    copy.add_target_indirect(store2);
    copy.add_output(store3);
    EXPECT_THROW(runtime->submit(std::move(copy)), std::runtime_error);
  }

  {
    auto copy = runtime->create_copy();
    copy.add_input(store1);
    copy.add_source_indirect(store2);
    copy.add_target_indirect(store3);
    copy.add_output(store1);
    EXPECT_THROW(runtime->submit(std::move(copy)), std::runtime_error);
  }
}

TEST(Copy, FailureInputOutputMismatch) { test_input_output_failure(); }

TEST(Copy, FailureMultipleIndirectCopies) { test_indirect_failure(); }

TEST(Copy, FailureDifferentShapes) { test_shape_check_failure(); }

}  // namespace copy_failure
