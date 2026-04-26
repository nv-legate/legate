/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/small_vector.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <utilities/utilities.h>

namespace small_vector_test {

namespace {

constexpr std::uint32_t SMALL_SIZE = 6;
constexpr std::int32_t BIG_COUNT   = 10;

using SmallVectorUnit = DefaultFixture;

}  // namespace

TEST_F(SmallVectorUnit, EraseRangeBigStorage)
{
  legate::detail::SmallVector<std::int32_t, SMALL_SIZE> vec;

  // Push past the inline capacity so the variant switches to big (std::vector) storage.
  vec.reserve(BIG_COUNT);
  for (std::int32_t i = 0; i < BIG_COUNT; ++i) {
    vec.push_back(i);
  }
  ASSERT_GT(vec.size(), SMALL_SIZE);

  // Erase a sub-range [2, 5): removes elements {2, 3, 4}.
  const auto it = vec.erase(vec.begin() + 2, vec.begin() + 5);

  ASSERT_EQ(vec.size(), BIG_COUNT - 3);
  ASSERT_EQ(*it, 5);
  const std::int32_t expected[] = {0, 1, 5, 6, 7, 8, 9};
  for (std::size_t i = 0; i < vec.size(); ++i) {
    ASSERT_EQ(vec[i], expected[i]);
  }
}

TEST_F(SmallVectorUnit, AtOutOfRangeThrows)
{
  legate::detail::SmallVector<std::int32_t, SMALL_SIZE> vec;

  vec.push_back(1);
  vec.push_back(2);
  ASSERT_THAT(
    [&] { (void)vec.at(vec.size()); },
    ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr("inplace_vector::at")));
}

TEST_F(SmallVectorUnit, ReserveOnBigStorage)
{
  legate::detail::SmallVector<std::int32_t, SMALL_SIZE> vec;

  // Promote to big storage so the reserve call hits the big_storage_type lambda.
  vec.reserve(BIG_COUNT);
  for (std::int32_t i = 0; i < BIG_COUNT; ++i) {
    vec.push_back(i);
  }
  ASSERT_GT(vec.size(), SMALL_SIZE);

  const auto cap_before = vec.capacity();
  vec.reserve(cap_before * 2);

  ASSERT_GE(vec.capacity(), cap_before * 2);
}

TEST_F(SmallVectorUnit, InsertOnBigStorage)
{
  legate::detail::SmallVector<std::int32_t, SMALL_SIZE> vec;

  // Promote to big storage so the insert call hits the big_storage_type lambda.
  vec.reserve(BIG_COUNT);
  for (std::int32_t i = 0; i < BIG_COUNT; ++i) {
    vec.push_back(i);
  }
  ASSERT_GT(vec.size(), SMALL_SIZE);

  const auto size_before = vec.size();
  const auto it          = vec.insert(vec.begin() + 2, 99);

  ASSERT_EQ(vec.size(), size_before + 1);
  ASSERT_EQ(*it, 99);
  ASSERT_EQ(vec[0], 0);
  ASSERT_EQ(vec[1], 1);
  ASSERT_EQ(vec[2], 99);
  ASSERT_EQ(vec[3], 2);
}

}  // namespace small_vector_test
