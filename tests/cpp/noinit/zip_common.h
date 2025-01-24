/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gtest/gtest.h>

#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <type_traits>
#include <vector>

namespace zip_iterator_common {

// NOLINTBEGIN(readability-magic-numbers)

template <typename ZipFunction>
class ZipTester {
  static constexpr ZipFunction ZIPPER_FN{};

  template <typename T>
  [[nodiscard]] static std::
    tuple<std::forward_list<float>, std::list<double>, std::map<int, double>>
    create_containers_from_reference_(const std::vector<T>& ref,
                                      std::optional<std::size_t> expected_size)
  {
    std::forward_list<float> flist;
    std::list<double> list;
    std::map<int, double> map;

    auto flit = flist.before_begin();
    T val{};
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

 public:
  static void modify_test_base(std::vector<int> base,
                               std::optional<std::size_t> expected_size = std::nullopt)
  {
    const auto expect_count = base.size();
    auto [x, y, z]          = create_containers_from_reference_(base, std::move(expected_size));
    auto count              = 0;
    auto zipper             = ZIPPER_FN(base, x, y, z);
    using pair_type         = typename std::decay_t<decltype(z)>::value_type;
    using count_type        = std::decay_t<decltype(count)>;

    for (auto&& [bi, xi, yi, zi] : zipper) {
      ++count;
      ASSERT_EQ(bi, count);
      bi = -count;
      ASSERT_EQ(xi, static_cast<float>(count));
      xi = -static_cast<float>(count);
      ASSERT_EQ(yi, static_cast<double>(count));
      yi = -static_cast<double>(count);
      ASSERT_EQ(zi, pair_type(count, static_cast<double>(count)));
      zi.second = -static_cast<double>(count);
    }
    ASSERT_EQ(count, expect_count);

    const auto compare_simple_container = [&](const auto& container) {
      count = 0;
      for (const auto& cit : container) {
        ++count;
        if (count <= static_cast<count_type>(base.size())) {
          // values modified in loop above
          ASSERT_EQ(cit, -count);
        } else {
          // the reference should never get here, since the iteration above is based on its size,
          // and so it should only ever have negative values
          ASSERT_NE((void*)std::addressof(container), (void*)std::addressof(base));
          // values untouched by loop
          ASSERT_EQ(cit, count);
        }
      }
    };

    compare_simple_container(base);
    compare_simple_container(x);
    compare_simple_container(y);
    // special treatment for map, since it has 2 elems per iterator
    count = 0;
    for (const auto& [_, zi] : z) {
      ++count;
      if (count <= static_cast<count_type>(base.size())) {
        ASSERT_EQ(zi, -count);
      } else {
        ASSERT_EQ(zi, count);
      }
    }
  }

  static void construct_test()
  {
    // These are mostly to test that the code compiles
    {
      std::vector<int> x;
      std::vector<int> y;

      auto zip = ZIPPER_FN(x, y);
      // so that zip is not optimized away
      ASSERT_EQ(zip.begin(), zip.end());
    }

    {
      std::vector<int> x;
      std::vector<double> y;

      auto zip = ZIPPER_FN(x, y);
      // so that zip is not optimized away
      ASSERT_EQ(zip.begin(), zip.end());
    }

    {
      std::list<int> x;
      std::vector<int> y;

      auto zip = ZIPPER_FN(x, y);
      // so that zip is not optimized away
      ASSERT_EQ(zip.begin(), zip.end());
    }

    {
      std::list<double> x;
      std::vector<int> y;

      auto zip = ZIPPER_FN(x, y);
      // so that zip is not optimized away
      ASSERT_EQ(zip.begin(), zip.end());
    }

    {
      std::list<double> x;
      std::forward_list<int> y;

      auto zip = ZIPPER_FN(x, y);
      // so that zip is not optimized away
      ASSERT_EQ(zip.begin(), zip.end());
    }
  }

  static void empty_test()
  {
    std::forward_list<float> w;
    std::vector<int> x;
    std::list<double> y;
    std::map<int, double> z;
    auto count  = 0;
    auto zipper = ZIPPER_FN(w, x, y, z);

    // the containers are empty, these loops should be a no-op
    for (auto it = zipper.begin(); it != zipper.end(); ++it) {
      ++count;
    }
    ASSERT_EQ(count, 0);
    for (auto it : zipper) {
      ++count;
      (void)it;
    }
    ASSERT_EQ(count, 0);
  }

  static void same_size_test()
  {
    std::forward_list<float> w{1.0, 2.0, 3.0, 4.0};
    std::vector<int> x{1, 2, 3, 4};
    std::list<double> y{1.0, 2.0, 3.0, 4.0};
    std::map<int, double> z{{1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0}};
    using pair_type = std::decay_t<decltype(z)>::value_type;

    const auto expect_count = x.size();
    auto count              = 0;
    auto zipper             = ZIPPER_FN(w, x, y, z);

    // the containers are all the same size, count should be equal to container sizes
    for (auto it = zipper.begin(); it != zipper.end(); ++it) {
      ++count;
      auto&& [wi, xi, yi, zi] = *it;
      ASSERT_EQ(wi, static_cast<float>(count));
      ASSERT_EQ(xi, count);
      ASSERT_EQ(yi, static_cast<double>(count));
      ASSERT_EQ(zi, pair_type(count, static_cast<double>(count)));
    }
    ASSERT_EQ(count, expect_count);

    count = 0;
    for (auto&& [wi, xi, yi, zi] : zipper) {
      ++count;
      ASSERT_EQ(wi, static_cast<float>(count));
      ASSERT_EQ(xi, count);
      ASSERT_EQ(yi, static_cast<double>(count));
      ASSERT_EQ(zi, pair_type(count, static_cast<double>(count)));
    }
    ASSERT_EQ(count, expect_count);
  }

  static void all_same_size_modify_test() { modify_test_base({1, 2, 3, 4}); }

  static void random_access_test()
  {
    std::vector<int> x{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto y = x;
    std::deque<double> z{x.begin(), x.end()};

    auto zipper = ZIPPER_FN(x, y, z);
    auto it     = zipper.begin();

    const auto assert_sane_iterator = [&](int inc) {
      ASSERT_EQ(it - zipper.begin(), inc);
      ASSERT_EQ(zipper.begin() - it, -inc);
      ASSERT_EQ(zipper.begin() + inc, it);
      ASSERT_EQ(it - inc, zipper.begin());
      ASSERT_EQ(*(zipper.begin() + inc), *it);
      ASSERT_EQ(*zipper.begin(), *(it - inc));
      if (inc) {
        ASSERT_FALSE(it < zipper.begin());
        ASSERT_TRUE(zipper.begin() < it);
        ASSERT_TRUE(it > zipper.begin());
        ASSERT_FALSE(zipper.begin() > it);
        ASSERT_FALSE(it <= zipper.begin());
        ASSERT_TRUE(zipper.begin() <= it);
        ASSERT_TRUE(it >= zipper.begin());
        ASSERT_FALSE(zipper.begin() >= it);
      } else {
        ASSERT_EQ(zipper.begin(), it);
        ASSERT_EQ(*zipper.begin(), *it);
        ASSERT_FALSE(it < zipper.begin());
        ASSERT_FALSE(zipper.begin() < it);
        ASSERT_FALSE(it > zipper.begin());
        ASSERT_FALSE(zipper.begin() > it);
        ASSERT_TRUE(it <= zipper.begin());
        ASSERT_TRUE(zipper.begin() <= it);
        ASSERT_TRUE(it >= zipper.begin());
        ASSERT_TRUE(zipper.begin() >= it);
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

  static void relational_test()
  {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {11, 12, 13, 14, 15};
    auto zipper        = ZIPPER_FN(a, b);
    auto it            = zipper.begin();

    for (std::size_t i = 0; i < a.size() - 1; ++i) {
      ASSERT_TRUE(it < zipper.end());
      ASSERT_FALSE(it > zipper.end());
      ASSERT_TRUE(it != zipper.end());
      ++it;
    }
    ASSERT_TRUE(it < zipper.end());
    ASSERT_FALSE(it > zipper.end());
    ASSERT_TRUE(it != zipper.end());
    ++it;
    ASSERT_FALSE(it < zipper.end());
    ASSERT_FALSE(it > zipper.end());
    ASSERT_TRUE(it == zipper.end());
  }
};

// NOLINTEND(readability-magic-numbers)

}  // namespace zip_iterator_common
