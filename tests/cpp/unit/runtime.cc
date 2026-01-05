/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/runtime.h>

#include <legate.h>

#include <legate/data/detail/shape.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_runtime {

using Runtime = DefaultFixture;

TEST_F(Runtime, CreateUnbound)
{
  constexpr std::uint32_t dim = 10;
  auto shape                  = legate::make_internal_shared<legate::detail::Shape>(dim);
  auto runtime                = legate::Runtime::get_runtime();
  ASSERT_THROW(static_cast<void>(runtime->create_array(legate::Shape{shape}, legate::int64())),
               std::invalid_argument);
  ASSERT_THROW(static_cast<void>(runtime->create_store(legate::Shape{shape}, legate::int64())),
               std::invalid_argument);
}

TEST_F(Runtime, IndexSpaceTooManyDims)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto runtime_impl = runtime->impl();
  auto data         = std::array<std::uint64_t, LEGATE_MAX_DIM + 1>{};

  data.fill(1);

  const auto span = legate::Span<const std::uint64_t>{data};

  ASSERT_THAT(
    [&] { static_cast<void>(runtime_impl->find_or_create_index_space(span)); },
    ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr(fmt::format(
      "Legate is configured with the maximum number of dimensions set to {}, but got a {}-D shape",
      LEGATE_MAX_DIM,
      data.size()))));
}

}  // namespace test_runtime
