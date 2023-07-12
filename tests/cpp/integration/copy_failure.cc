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

void test_invalid_stores()
{
  auto runtime = legate::Runtime::get_runtime();

  auto store1 = runtime->create_store({10, 10}, legate::int64());
  auto store2 = runtime->create_store({1}, legate::int64(), true /*optimize_scalar*/);
  auto store3 = runtime->create_store(legate::int64());
  auto store4 = runtime->create_store({10, 10}, legate::int64()).promote(2, 10);

  EXPECT_THROW(runtime->issue_copy(store2, store1), std::invalid_argument);
  EXPECT_THROW(runtime->issue_copy(store3, store1), std::invalid_argument);
  EXPECT_THROW(runtime->issue_copy(store4, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_gather(store2, store3, store1), std::invalid_argument);
  EXPECT_THROW(runtime->issue_gather(store3, store4, store1), std::invalid_argument);
  EXPECT_THROW(runtime->issue_gather(store4, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(store2, store3, store1), std::invalid_argument);
  EXPECT_THROW(runtime->issue_scatter(store3, store4, store1), std::invalid_argument);
  EXPECT_THROW(runtime->issue_scatter(store4, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(store2, store3, store4, store1),
               std::invalid_argument);
  EXPECT_THROW(runtime->issue_scatter_gather(store3, store4, store2, store1),
               std::invalid_argument);
  EXPECT_THROW(runtime->issue_scatter_gather(store4, store2, store3, store1),
               std::invalid_argument);
}

void test_type_check_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto source          = runtime->create_store({10, 10}, legate::int64());
  auto target          = runtime->create_store({10, 10}, legate::float64());
  auto source_indirect = runtime->create_store({10, 10}, legate::point_type(2));
  auto target_indirect = runtime->create_store({10, 10}, legate::point_type(2));

  EXPECT_THROW(runtime->issue_copy(target, source), std::invalid_argument);

  EXPECT_THROW(runtime->issue_gather(target, source, source_indirect), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(target, target_indirect, source), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(target, target_indirect, source, source_indirect),
               std::invalid_argument);
}

void test_shape_check_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto store1 = runtime->create_store({10, 10}, legate::int64());
  auto store2 = runtime->create_store({5, 20}, legate::int64());
  auto store3 = runtime->create_store({20, 5}, legate::int64());
  auto store4 = runtime->create_store({5, 5}, legate::int64());

  EXPECT_THROW(runtime->issue_copy(store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_gather(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(store4, store3, store2, store1),
               std::invalid_argument);
}

void test_non_point_types_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto store1 = runtime->create_store({10, 10}, legate::int32());
  auto store2 = runtime->create_store({10, 10}, legate::int32());
  auto store3 = runtime->create_store({10, 10}, legate::int32());
  auto store4 = runtime->create_store({10, 10}, legate::int32());

  EXPECT_THROW(runtime->issue_gather(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(store4, store3, store2, store1),
               std::invalid_argument);
}

void test_dimension_mismatch_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto source          = runtime->create_store({10, 10}, legate::int64());
  auto target          = runtime->create_store({10, 10}, legate::int64());
  auto source_indirect = runtime->create_store({10, 10}, legate::point_type(3));
  auto target_indirect = runtime->create_store({10, 10}, legate::point_type(3));

  EXPECT_THROW(runtime->issue_gather(target, source, source_indirect), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(target, target_indirect, source), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(target, target_indirect, source, source_indirect),
               std::invalid_argument);
}

TEST(Copy, FailureInvalidStores) { test_invalid_stores(); }

TEST(Copy, FailureDifferentTypes) { test_type_check_failure(); }

TEST(Copy, FailureDifferentShapes) { test_shape_check_failure(); }

TEST(Copy, FailureNonPointTypes) { test_non_point_types_failure(); }

TEST(Copy, FailureDimensionMismatch) { test_dimension_mismatch_failure(); }

}  // namespace copy_failure
