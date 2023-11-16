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

#include "core/utilities/detail/zip.h"

#include <deque>
#include <forward_list>
#include <gtest/gtest.h>
#include <list>
#include <map>
#include <type_traits>
#include <vector>

TEST(ZipIteratorUnit, Construct)
{
  // These are mostly to test that the code compiles
  {
    std::vector<int> x;
    std::vector<int> y;

    auto zip = legate::detail::zip(x, y);
    // so that zip is not optimized away
    EXPECT_EQ(zip.begin(), zip.end());
  }

  {
    std::vector<int> x;
    std::vector<double> y;

    auto zip = legate::detail::zip(x, y);
    // so that zip is not optimized away
    EXPECT_EQ(zip.begin(), zip.end());
  }

  {
    std::list<int> x;
    std::vector<int> y;

    auto zip = legate::detail::zip(x, y);
    // so that zip is not optimized away
    EXPECT_EQ(zip.begin(), zip.end());
  }

  {
    std::list<double> x;
    std::vector<int> y;

    auto zip = legate::detail::zip(x, y);
    // so that zip is not optimized away
    EXPECT_EQ(zip.begin(), zip.end());
  }

  {
    std::list<double> x;
    std::forward_list<int> y;

    auto zip = legate::detail::zip(x, y);
    // so that zip is not optimized away
    EXPECT_EQ(zip.begin(), zip.end());
  }
}

namespace {

template <typename T>
[[nodiscard]] std::tuple<std::forward_list<float>, std::list<double>, std::map<int, double>>
create_containers_from_reference(const std::vector<T>& ref,
                                 std::optional<std::size_t> expected_size)
{
  std::forward_list<float> flist;
  std::list<double> list;
  std::map<int, double> map;

  auto flit = flist.before_begin();
  T val;
  for (std::size_t i = 0; i < expected_size.value_or(ref.size()); ++i) {
    if (i < ref.size()) {
      val = ref.at(i);
    } else {
      ++val;
    }
    flit = flist.emplace_after(flit, val);
    list.emplace_back(val);
    map.emplace(val, val);
  }
  return {flist, list, map};
}

}  // namespace

TEST(ZipIteratorUnit, IterateEmpty)
{
  std::forward_list<float> w;
  std::vector<int> x;
  std::list<double> y;
  std::map<int, double> z;
  auto count  = 0;
  auto zipper = legate::detail::zip(w, x, y, z);

  // the containers are empty, these loops should be a no-op
  for (auto it = zipper.begin(); it != zipper.end(); ++it) {
    ++count;
  }
  EXPECT_EQ(count, 0);
  for (auto it : zipper) {
    ++count;
    (void)it;
  }
  EXPECT_EQ(count, 0);
}

TEST(ZipIteratorUnit, IterateAllSameSize)
{
  std::forward_list<float> w{1.0, 2.0, 3.0, 4.0};
  std::vector<int> x{1, 2, 3, 4};
  std::list<double> y{1.0, 2.0, 3.0, 4.0};
  std::map<int, double> z{{1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0}};
  using pair_type = std::decay_t<decltype(z)>::value_type;

  const auto expect_count = x.size();
  auto count              = 0;
  auto zipper             = legate::detail::zip(w, x, y, z);

  // the containers are all the same size, count should be equal to container sizes
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

TEST(ZipIteratorUnit, IterateOneShort)
{
  std::forward_list<float> w{1.0, 2.0, 3.0, 4.0};
  std::vector<int> x{1, 2, 3};  // NOTE vector is shorter than the rest!
  std::list<double> y{1.0, 2.0, 3.0, 4.0};
  std::map<int, double> z{{1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0}};
  using pair_type = std::decay_t<decltype(z)>::value_type;

  const auto expect_count = x.size();
  auto count              = 0;
  auto zipper             = legate::detail::zip(w, x, y, z);

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

namespace {

void ModifyTestBase(std::vector<int> base, std::optional<std::size_t> expected_size = std::nullopt)
{
  const auto expect_count = base.size();
  auto [x, y, z]          = create_containers_from_reference(base, std::move(expected_size));
  auto count              = 0;
  auto zipper             = legate::detail::zip(base, x, y, z);
  using pair_type         = std::decay_t<decltype(z)>::value_type;

  for (auto&& [bi, xi, yi, zi] : zipper) {
    ++count;
    EXPECT_EQ(bi, count);
    bi = -count;
    EXPECT_EQ(xi, static_cast<float>(count));
    xi = -static_cast<float>(count);
    EXPECT_EQ(yi, static_cast<double>(count));
    yi = -static_cast<double>(count);
    EXPECT_EQ(zi, pair_type(count, static_cast<double>(count)));
    zi.second = -static_cast<double>(count);
  }
  EXPECT_EQ(count, expect_count);

  const auto compare_simple_container = [&](const auto& container) {
    count = 0;
    for (const auto& cit : container) {
      if (++count <= static_cast<decltype(count)>(base.size())) {
        // values modified in loop above
        EXPECT_EQ(cit, -count);
      } else {
        // the reference should never get here, since the iteration above is based on its size,
        // and so it should only ever have negative values
        EXPECT_NE((void*)std::addressof(container), (void*)std::addressof(base));
        // values untoched by loop
        EXPECT_EQ(cit, count);
      }
    }
  };

  compare_simple_container(base);
  compare_simple_container(x);
  compare_simple_container(y);
  // special treatment for map, since it has 2 elems per iterator
  count = 0;
  for (const auto& [_, zi] : z) {
    if (++count <= static_cast<decltype(count)>(base.size())) {
      EXPECT_EQ(zi, -count);
    } else {
      EXPECT_EQ(zi, count);
    }
  }
}

}  // namespace

TEST(ZipIteratorUnit, IterateAllSameSizeModify) { ModifyTestBase({1, 2, 3, 4}); }

TEST(ZipIteratorUnit, IterateOneShortModify) { ModifyTestBase({1, 2, 3}, {10}); }

TEST(ZipIteratorUnit, RandomAccess)
{
  std::vector<int> x{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto y = x;
  std::deque<double> z{x.begin(), x.end()};

  auto zipper = legate::detail::zip(x, y, z);
  auto it     = zipper.begin();

  const auto assert_sane_iterator = [&](int inc) {
    EXPECT_EQ(it - zipper.begin(), inc);
    EXPECT_EQ(zipper.begin() - it, -inc);
    EXPECT_EQ(zipper.begin() + inc, it);
    EXPECT_EQ(it - inc, zipper.begin());
    EXPECT_EQ(*(zipper.begin() + inc), *it);
    EXPECT_EQ(*zipper.begin(), *(it - inc));
    if (inc) {
      EXPECT_FALSE(it < zipper.begin());
      EXPECT_TRUE(zipper.begin() < it);
      EXPECT_TRUE(it > zipper.begin());
      EXPECT_FALSE(zipper.begin() > it);
      EXPECT_FALSE(it <= zipper.begin());
      EXPECT_TRUE(zipper.begin() <= it);
      EXPECT_TRUE(it >= zipper.begin());
      EXPECT_FALSE(zipper.begin() >= it);
    } else {
      EXPECT_EQ(zipper.begin(), it);
      EXPECT_EQ(*zipper.begin(), *it);
      EXPECT_FALSE(it < zipper.begin());
      EXPECT_FALSE(zipper.begin() < it);
      EXPECT_FALSE(it > zipper.begin());
      EXPECT_FALSE(zipper.begin() > it);
      EXPECT_TRUE(it <= zipper.begin());
      EXPECT_TRUE(zipper.begin() <= it);
      EXPECT_TRUE(it >= zipper.begin());
      EXPECT_TRUE(zipper.begin() >= it);
    }
  };

  static_assert(std::is_convertible_v<typename std::decay_t<decltype(zipper)>::iterator_category,
                                      std::random_access_iterator_tag>);

  assert_sane_iterator(0);
  it += 1;
  assert_sane_iterator(1);
  ++it;
  assert_sane_iterator(2);
  it -= 1;
  assert_sane_iterator(1);
  --it;
  assert_sane_iterator(0);
  it += 8;
  assert_sane_iterator(8);
  it -= 4;
  assert_sane_iterator(4);
  ++it;
  assert_sane_iterator(5);
  it -= 5;
  assert_sane_iterator(0);
}

TEST(ZipIteratorUnit, Relational)
{
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> b = {11, 12, 13};
  auto zipper        = legate::detail::zip(a, b);
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

TEST(ZipIteratorUnit, DoxySnippets)
{
  /// [Constructing a zipper]
  std::vector<float> vec{1, 2, 3, 4, 5};
  std::list<int> list{5, 4, 3, 2, 1};

  // Add all elements of a list to each element of a vector
  for (auto&& [vi, li] : legate::detail::zip(vec, list)) {
    vi = li + 10;
    std::cout << vi << ", ";
  }
  /// [Constructing a zipper]
}
