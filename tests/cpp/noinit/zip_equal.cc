/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/zip.h>

#include <gtest/gtest.h>

#include <noinit/zip_common.h>
#include <utility>

namespace zip_longest_test {

namespace {

class ZipEqualFn {
 public:
  template <typename... T>
  [[nodiscard]] auto operator()(T&&... args) const
  {
    return legate::detail::zip_equal(std::forward<T>(args)...);
  }
};

using ZipTester = zip_iterator_common::ZipTester<ZipEqualFn>;

}  // namespace

namespace has_size_test {

static_assert(
  legate::detail::is_detected_v<legate::detail::zip_detail::has_size, std::vector<std::int32_t>>);
static_assert(!legate::detail::is_detected_v<legate::detail::zip_detail::has_size, std::int32_t>);

static_assert(
  std::conjunction_v<
    legate::detail::is_detected<legate::detail::zip_detail::has_size, std::vector<std::int32_t>>,
    legate::detail::is_detected<legate::detail::zip_detail::has_size, std::vector<std::int32_t>>>);

}  // namespace has_size_test

TEST(ZipEqual, BadSize)
{
  const std::vector<std::int32_t> v1{1, 2, 3, 4};
  const std::vector<std::int32_t> v2{1, 2, 3};

  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Throwing check is only performed in debug mode
    ASSERT_THROW(static_cast<void>(legate::detail::zip_equal(v1, v2)), std::invalid_argument);
  } else {
    ASSERT_NO_THROW(static_cast<void>(legate::detail::zip_equal(v1, v2)));
  }
}

TEST(ZipEqualUnit, Construct) { ZipTester::construct_test(); }

TEST(ZipEqualUnit, IterateEmpty) { ZipTester::empty_test(); }

TEST(ZipEqualUnit, IterateAllSameSize) { ZipTester::same_size_test(); }

TEST(ZipEqualUnit, IterateAllSameSizeModify) { ZipTester::all_same_size_modify_test(); }

TEST(ZipEqualUnit, RandomAccess) { ZipTester::random_access_test(); }

TEST(ZipEqualUnit, Relational) { ZipTester::relational_test(); }

TEST(ZipEqualUnit, DoxySnippets)
{
  // NOLINTBEGIN(readability-magic-numbers)
  /// [Constructing a zipper]
  std::vector<float> vec{1, 2, 3, 4, 5};
  std::list<int> list{5, 4, 3, 2, 1};

  // Add all elements of a list to each element of a vector
  for (auto&& [vi, li] : legate::detail::zip_equal(vec, list)) {
    vi = static_cast<float>(li + 10);
    std::cout << vi << ", ";
  }
  /// [Constructing a zipper]
  // NOLINTEND(readability-magic-numbers)
}

}  // namespace zip_longest_test
