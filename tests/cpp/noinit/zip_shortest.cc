/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/utilities/detail/zip.h>

#include <gtest/gtest.h>

#include <forward_list>
#include <list>
#include <map>
#include <noinit/zip_common.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace zip_shortest_test {

// NOLINTBEGIN(readability-magic-numbers)

class ZipShortestFn {
 public:
  template <typename... T>
  [[nodiscard]] auto operator()(T&&... args) const
  {
    return legate::detail::zip_shortest(std::forward<T>(args)...);
  }
};

using ZipTester = zip_iterator_common::ZipTester<ZipShortestFn>;

TEST(ZipShortestUnit, Construct) { ZipTester::construct_test(); }

TEST(ZipShortestUnit, IterateEmpty) { ZipTester::empty_test(); }

TEST(ZipShortestUnit, IterateAllSameSize) { ZipTester::same_size_test(); }

TEST(ZipShortestUnit, IterateOneShort)
{
  std::forward_list<float> w{1.0, 2.0, 3.0, 4.0};
  std::vector<int> x{1, 2, 3};  // NOTE vector is shorter than the rest!
  std::list<double> y{1.0, 2.0, 3.0, 4.0};
  std::map<int, double> z{{1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0}};
  using pair_type = std::decay_t<decltype(z)>::value_type;

  const auto expect_count = x.size();
  auto count              = 0;
  auto zipper             = legate::detail::zip_shortest(w, x, y, z);

  // count should be equal to smallest container size
  for (auto it = zipper.begin(); it != zipper.end(); ++it) {
    ++count;
    auto&& [wi, xi, yi, zi] = *it;
    EXPECT_EQ(wi, static_cast<float>(count));
    EXPECT_EQ(xi, count);
    EXPECT_EQ(yi, static_cast<double>(count));
    EXPECT_EQ(zi, pair_type(count, static_cast<double>(count)));
  }
  EXPECT_EQ(count, expect_count);

  count = 0;
  for (auto&& [wi, xi, yi, zi] : zipper) {
    ++count;
    EXPECT_EQ(wi, static_cast<float>(count));
    EXPECT_EQ(xi, count);
    EXPECT_EQ(yi, static_cast<double>(count));
    EXPECT_EQ(zi, pair_type(count, static_cast<double>(count)));
  }
  EXPECT_EQ(count, expect_count);
}

TEST(ZipShortestUnit, IterateAllSameSizeModify) { ZipTester::all_same_size_modify_test(); }

TEST(ZipShortestUnit, IterateOneShortModify) { ZipTester::modify_test_base({1, 2, 3}, {10}); }

TEST(ZipShortestUnit, RandomAccess) { ZipTester::random_access_test(); }

TEST(ZipShortestUnit, Relational)
{
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> b = {11, 12, 13};
  auto zipper        = legate::detail::zip_shortest(a, b);
  auto it            = zipper.begin();
  it += 3;
  // only one of these should be true (or none):
  EXPECT_EQ(it, zipper.end());  // because std::get<1>(it.iters()) is at the end of `b`
  EXPECT_GE(it, zipper.end());
  EXPECT_LE(it, zipper.end());
  EXPECT_FALSE(it < zipper.end());
  EXPECT_FALSE(it > zipper.end());
  EXPECT_FALSE(it != zipper.end());
}

TEST(ZipShortestUnit, DoxySnippets)
{
  /// [Constructing a zipper]
  std::vector<float> vec{1, 2, 3, 4, 5};
  std::list<int> list{5, 4, 3, 2, 1};

  // Add all elements of a list to each element of a vector
  for (auto&& [vi, li] : legate::detail::zip_equal(vec, list)) {
    vi = static_cast<float>(li + 10);
    std::cout << vi << ", ";
  }
  /// [Constructing a zipper]
}

// NOLINTEND(readability-magic-numbers)

}  // namespace zip_shortest_test
