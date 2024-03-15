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

#include "core/utilities/detail/enumerate.h"

#include <cstddef>
#include <deque>
#include <gtest/gtest.h>
#include <list>
#include <tuple>
#include <vector>

namespace {

template <typename Container, typename T>
void TestContainer(const std::vector<T>& init_values)
{
  auto container              = Container{init_values.begin(), init_values.end()};
  const auto size             = static_cast<std::ptrdiff_t>(init_values.size());
  const auto backup_container = container;

  for (std::ptrdiff_t start = 0; start < size; ++start) {
    auto backup_it  = backup_container.begin();
    auto backup_idx = start;

    for (auto&& [idx, val] : legate::detail::enumerate(container, start)) {
      EXPECT_EQ(backup_idx, idx);
      EXPECT_NE(backup_it, backup_container.end());
      EXPECT_EQ(val, *backup_it);
      ++backup_it;
      ++backup_idx;
    }
    EXPECT_EQ(backup_idx, init_values.size() + start);
  }
}

}  // namespace

using EnumerateTypeList =
  ::testing::Types<std::vector<int>, std::list<float>, std::deque<std::int64_t>>;

template <typename>
struct EnumerateUnit : ::testing::Test {};

TYPED_TEST_SUITE(EnumerateUnit, EnumerateTypeList, );

TYPED_TEST(EnumerateUnit, Basic)
{
  std::vector<typename TypeParam::value_type> init_values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  TestContainer<TypeParam>(init_values);
}

TYPED_TEST(EnumerateUnit, Bidirectional)
{
  std::vector<typename TypeParam::value_type> init_values{1, 2};

  auto enumerator = legate::detail::enumerate(init_values);
  auto it         = enumerator.begin();
  auto val_it     = init_values.begin();

  EXPECT_NE(it, enumerator.end());
  EXPECT_EQ(*it, std::make_tuple(0, *val_it));
  ++it;
  EXPECT_NE(it, enumerator.end());
  EXPECT_EQ(*it, std::make_tuple(1, *(val_it + 1)));
  --it;
  EXPECT_EQ(*it, std::make_tuple(0, *val_it));
}

TEST(EnumerateUnit, DoxySnippets)
{
  /// [Constructing an enumerator]
  std::vector<int> my_vector{1, 2, 3, 4, 5};

  // Enumerate a vector starting from index 0
  for (auto&& [idx, val] : legate::detail::enumerate(my_vector)) {
    std::cout << "accessing element " << idx << " of vector: " << val << '\n';
    // a sanity check
    EXPECT_EQ(my_vector[idx], val);
  }

  // Enumerate the vector, but enumerator starts at index 3. Note that the enumerator start has
  // no bearing on the thing being enumerated. The vector is still iterated over from start to
  // finish!
  auto enum_start = 3;
  for (auto&& [idx, val] : legate::detail::enumerate(my_vector, enum_start)) {
    std::cout << "enumerator has value: " << idx << '\n';
    std::cout << "accessing element " << idx - enum_start << " of vector: " << val << '\n';
    EXPECT_EQ(my_vector[idx - enum_start], val);
  }
  /// [Constructing an enumerator]
}
