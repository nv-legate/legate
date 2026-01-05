/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/future_wrapper.h>

#include <legate.h>

#include <legate/data/detail/logical_store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace future_wrapper_test {

using FutureWrapperTest = DefaultFixture;

TEST_F(FutureWrapperTest, Empty)
{
  auto future_wrapper = legate::detail::FutureWrapper{};

  ASSERT_FALSE(future_wrapper.valid());
  ASSERT_TRUE(future_wrapper.is_read_only());
  ASSERT_EQ(future_wrapper.dim(), 0);
  ASSERT_EQ(future_wrapper.domain().dim, 0);
  ASSERT_EQ(future_wrapper.field_size(), 0);
  ASSERT_EQ(future_wrapper.field_offset(), 0);
  ASSERT_EQ(future_wrapper.target(), legate::mapping::StoreTarget::SYSMEM);
}

TEST_F(FutureWrapperTest, Readonly)
{
  auto runtime                      = legate::Runtime::get_runtime();
  constexpr std::int64_t TEST_VALUE = 42;
  auto scalar                       = legate::Scalar{TEST_VALUE};
  auto logical_store                = runtime->create_store(scalar);
  auto future                       = logical_store.impl()->get_future();

  ASSERT_TRUE(future.valid());

  auto domain                             = legate::Domain{};
  constexpr std::uint32_t FIELD_SIZE      = sizeof(std::int64_t);
  constexpr std::uint32_t FIELD_ALIGNMENT = alignof(std::int64_t);
  constexpr std::uint64_t FIELD_OFFSET    = 0;
  auto future_wrapper                     = legate::detail::FutureWrapper{
    /*read_only=*/true, FIELD_SIZE, FIELD_ALIGNMENT, FIELD_OFFSET, domain, std::move(future)};

  ASSERT_TRUE(future_wrapper.valid());
  ASSERT_EQ(future_wrapper.dim(), 0);
  ASSERT_EQ(future_wrapper.domain().dim, 0);
  ASSERT_EQ(future_wrapper.field_size(), FIELD_SIZE);
  ASSERT_EQ(future_wrapper.field_offset(), FIELD_OFFSET);
  ASSERT_TRUE(future_wrapper.is_read_only());
  ASSERT_EQ(future_wrapper.target(), legate::mapping::StoreTarget::SYSMEM);
  ASSERT_NO_THROW({
    const void* ptr = future_wrapper.get_untyped_pointer_from_future();
    ASSERT_NE(ptr, nullptr);
    const auto* typed_ptr = static_cast<const std::int64_t*>(ptr);
    ASSERT_EQ(*typed_ptr, TEST_VALUE);
  });
}

TEST_F(FutureWrapperTest, Writable)
{
  auto runtime                            = legate::Runtime::get_runtime();
  constexpr std::int32_t TEST_VALUE       = 123;
  auto scalar                             = legate::Scalar{TEST_VALUE};
  auto logical_store                      = runtime->create_store(scalar);
  auto future                             = logical_store.impl()->get_future();
  auto domain                             = legate::Domain{};
  constexpr std::uint32_t FIELD_SIZE      = sizeof(std::int32_t);
  constexpr std::uint32_t FIELD_ALIGNMENT = alignof(std::int32_t);
  constexpr std::uint64_t FIELD_OFFSET    = 0;

  auto future_wrapper = legate::detail::FutureWrapper{
    /*read_only=*/false, FIELD_SIZE, FIELD_ALIGNMENT, FIELD_OFFSET, domain, std::move(future)};

  ASSERT_TRUE(future_wrapper.valid());
  ASSERT_FALSE(future_wrapper.is_read_only());
  ASSERT_EQ(future_wrapper.field_size(), FIELD_SIZE);
  ASSERT_NO_THROW({
    const auto& buffer = future_wrapper.get_buffer();
    ASSERT_TRUE(buffer.get_instance().exists());
  });
}

}  // namespace future_wrapper_test
