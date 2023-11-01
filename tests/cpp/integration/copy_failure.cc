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
#include "utilities/utilities.h"

namespace copy_failure {

using Copy = DefaultFixture;

void test_invalid_stores()
{
  auto runtime = legate::Runtime::get_runtime();

  auto store1 = runtime->create_store(legate::Shape{10, 10}, legate::int64());
  auto store2 = runtime->create_store(legate::Shape{1}, legate::int64(), true /*optimize_scalar*/);
  auto store3 = runtime->create_store(legate::int64());
  auto store4 = runtime->create_store(legate::Shape{10, 10}, legate::int64()).promote(2, 10);

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

  auto shape           = legate::Shape{10, 10};
  auto source          = runtime->create_store(shape, legate::int64());
  auto target          = runtime->create_store(shape, legate::float64());
  auto source_indirect = runtime->create_store(shape, legate::point_type(2));
  auto target_indirect = runtime->create_store(shape, legate::point_type(2));

  EXPECT_THROW(runtime->issue_copy(target, source), std::invalid_argument);

  EXPECT_THROW(runtime->issue_gather(target, source, source_indirect), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(target, target_indirect, source), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(target, target_indirect, source, source_indirect),
               std::invalid_argument);
}

void test_shape_check_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto store1 = runtime->create_store(legate::Shape{10, 10}, legate::int64());
  auto store2 = runtime->create_store(legate::Shape{5, 20}, legate::int64());
  auto store3 = runtime->create_store(legate::Shape{20, 5}, legate::int64());
  auto store4 = runtime->create_store(legate::Shape{5, 5}, legate::int64());

  EXPECT_THROW(runtime->issue_copy(store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_gather(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(store4, store3, store2, store1),
               std::invalid_argument);
}

void test_non_point_types_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto shape  = legate::Shape{10, 10};
  auto store1 = runtime->create_store(shape, legate::int32());
  auto store2 = runtime->create_store(shape, legate::int32());
  auto store3 = runtime->create_store(shape, legate::int32());
  auto store4 = runtime->create_store(shape, legate::int32());

  EXPECT_THROW(runtime->issue_gather(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(store3, store2, store1), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(store4, store3, store2, store1),
               std::invalid_argument);
}

void test_dimension_mismatch_failure()
{
  auto runtime = legate::Runtime::get_runtime();

  auto shape           = legate::Shape{10, 10};
  auto source          = runtime->create_store(shape, legate::int64());
  auto target          = runtime->create_store(shape, legate::int64());
  auto source_indirect = runtime->create_store(shape, legate::point_type(3));
  auto target_indirect = runtime->create_store(shape, legate::point_type(3));

  EXPECT_THROW(runtime->issue_gather(target, source, source_indirect), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter(target, target_indirect, source), std::invalid_argument);

  EXPECT_THROW(runtime->issue_scatter_gather(target, target_indirect, source, source_indirect),
               std::invalid_argument);
}

TEST_F(Copy, FailureInvalidStores) { test_invalid_stores(); }

TEST_F(Copy, FailureDifferentTypes) { test_type_check_failure(); }

TEST_F(Copy, FailureDifferentShapes) { test_shape_check_failure(); }

TEST_F(Copy, FailureNonPointTypes) { test_non_point_types_failure(); }

TEST_F(Copy, FailureDimensionMismatch) { test_dimension_mismatch_failure(); }

}  // namespace copy_failure
