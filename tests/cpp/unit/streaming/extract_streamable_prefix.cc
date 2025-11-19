/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/operation/detail/execution_fence.h>
#include <legate/operation/detail/mapping_fence.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/streaming/analysis.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_extract_streamable_prefix {

class ExtractStreamablePrefixUnit : public DefaultFixture {};

TEST_F(ExtractStreamablePrefixUnit, EmptyInput)
{
  std::deque<legate::InternalSharedPtr<legate::detail::Operation>> q;

  auto prefix = legate::detail::extract_streamable_prefix(&q);
  ASSERT_TRUE(q.empty());
  ASSERT_TRUE(prefix.empty());
}

TEST_F(ExtractStreamablePrefixUnit, SingleMappingFence)
{
  std::deque<legate::InternalSharedPtr<legate::detail::Operation>> q;

  auto& rt = legate::detail::Runtime::get_runtime();
  auto scope =
    legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::STRICT)};

  q.emplace_back(legate::make_internal_shared<legate::detail::MappingFence>(rt.new_op_id()));

  auto prefix = legate::detail::extract_streamable_prefix(&q);
  ASSERT_TRUE(q.empty());
  ASSERT_EQ(prefix.size(), 1);
  ASSERT_EQ(prefix.back()->kind(), legate::detail::Operation::Kind::MAPPING_FENCE);
}

TEST_F(ExtractStreamablePrefixUnit, SingleOtherTask)
{
  std::deque<legate::InternalSharedPtr<legate::detail::Operation>> q;

  auto& rt = legate::detail::Runtime::get_runtime();
  // create non-streamble operation inside a streaming scope
  auto scope =
    legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::STRICT)};

  q.emplace_back(
    legate::make_internal_shared<legate::detail::ExecutionFence>(rt.new_op_id(), false));
  ASSERT_TRUE(q.back()->parallel_policy().streaming());

  auto prefix = legate::detail::extract_streamable_prefix(&q);

  // prefix must end in mapping fence
  ASSERT_TRUE(q.empty());
  ASSERT_EQ(prefix.size(), 2);
  ASSERT_EQ(prefix.back()->kind(), legate::detail::Operation::Kind::MAPPING_FENCE);
  ASSERT_EQ(prefix.front()->kind(), legate::detail::Operation::Kind::EXECUTION_FENCE);
}

TEST_F(ExtractStreamablePrefixUnit, TwoNonStreambleTasksRelaxed)
{
  std::deque<legate::InternalSharedPtr<legate::detail::Operation>> q;

  auto& rt = legate::detail::Runtime::get_runtime();
  // create two non-streamble operations inside a streaming scope
  // Expect two prefixes of size 1 each
  auto scope =
    legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::RELAXED)};

  q.emplace_back(
    legate::make_internal_shared<legate::detail::ExecutionFence>(rt.new_op_id(), false));
  q.emplace_back(
    legate::make_internal_shared<legate::detail::ExecutionFence>(rt.new_op_id(), false));
  ASSERT_TRUE(q.back()->parallel_policy().streaming());

  auto prefix1 = legate::detail::extract_streamable_prefix(&q);

  // only one task should have been extracted
  ASSERT_EQ(q.size(), 1);
  // prefix must end in mapping fence
  ASSERT_EQ(prefix1.size(), 2);
  ASSERT_EQ(prefix1.back()->kind(), legate::detail::Operation::Kind::MAPPING_FENCE);
  ASSERT_EQ(prefix1.front()->kind(), legate::detail::Operation::Kind::EXECUTION_FENCE);

  auto prefix2 = legate::detail::extract_streamable_prefix(&q);

  ASSERT_TRUE(q.empty());
  ASSERT_EQ(prefix1.size(), 2);
  ASSERT_EQ(prefix1.back()->kind(), legate::detail::Operation::Kind::MAPPING_FENCE);
  ASSERT_EQ(prefix1.front()->kind(), legate::detail::Operation::Kind::EXECUTION_FENCE);
}

TEST_F(ExtractStreamablePrefixUnit, TwoNonStreambleTasks)
{
  std::deque<legate::InternalSharedPtr<legate::detail::Operation>> q;

  auto& rt = legate::detail::Runtime::get_runtime();
  // create two non-streamble operations inside a streaming scope
  // In STRICT mode, expect an exception to be thrown.
  auto scope =
    legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::STRICT)};

  q.emplace_back(
    legate::make_internal_shared<legate::detail::ExecutionFence>(rt.new_op_id(), false));
  q.emplace_back(
    legate::make_internal_shared<legate::detail::ExecutionFence>(rt.new_op_id(), false));
  ASSERT_TRUE(q.back()->parallel_policy().streaming());

  ASSERT_THAT([&]() { return legate::detail::extract_streamable_prefix(&q); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Streaming Scope cannot be streamed in one go")));
}

}  // namespace test_extract_streamable_prefix
