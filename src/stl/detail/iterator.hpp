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

#pragma once

#include "core/utilities/detail/compressed_pair.h"

#include "config.hpp"
#include "utility.hpp"

#include <iterator>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

namespace detail {

template <typename Map>
using reference_t = decltype(std::declval<const Map&>().read(std::declval<typename Map::cursor>()));

template <typename Map, typename Iterator>
using mixin_ = typename Map::template mixin<Iterator>;

template <typename Map, typename Iterator>
using mixin = meta::eval<meta::quote_or<mixin_, meta::empty>, Map, Iterator>;

}  // namespace detail

template <typename Map>
class iterator : public detail::mixin<Map, iterator<Map>> {
 public:
  using difference_type   = std::ptrdiff_t;
  using value_type        = typename Map::value_type;
  using iterator_category = std::random_access_iterator_tag;
  using reference         = detail::reference_t<Map>;

  class pointer {
   public:
    value_type value_{};

    [[nodiscard]] value_type* operator->() && noexcept { return std::addressof(value_); }
  };

  iterator() = default;

  LEGATE_HOST_DEVICE iterator(Map map, typename Map::cursor cursor)
    : cursor_map_pair_{cursor, std::move(map)}
  {
  }

  LEGATE_HOST_DEVICE iterator& operator++()
  {
    cursor() = map().next(cursor());
    return *this;
  }

  LEGATE_HOST_DEVICE iterator operator++(int)
  {
    auto copy = *this;
    ++*this;
    return copy;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] reference operator*() const { return map().read(cursor()); }

  LEGATE_HOST_DEVICE [[nodiscard]] pointer operator->() const { return pointer{operator*()}; }

  LEGATE_HOST_DEVICE friend bool operator==(const iterator& lhs, const iterator& rhs)
  {
    return lhs.map().equal(lhs.cursor(), rhs.cursor());
  }

  LEGATE_HOST_DEVICE friend bool operator!=(const iterator& lhs, const iterator& rhs)
  {
    return !(lhs == rhs);
  }

  LEGATE_HOST_DEVICE iterator& operator--()
  {
    cursor() = map().prev(cursor());
    return *this;
  }

  LEGATE_HOST_DEVICE iterator operator--(int)
  {
    auto copy = *this;
    --*this;
    return copy;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] iterator operator+(difference_type n) const
  {
    return {map(), map().advance(cursor(), n)};
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend iterator operator+(difference_type n, const iterator& it)
  {
    return it + n;
  }

  LEGATE_HOST_DEVICE iterator& operator+=(difference_type n)
  {
    cursor() = map().advance(cursor(), n);
    return *this;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] iterator operator-(difference_type n) const
  {
    return {map(), map().advance(cursor(), -n)};
  }

  LEGATE_HOST_DEVICE iterator& operator-=(difference_type n)
  {
    cursor() = map().advance(cursor(), -n);
    return *this;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend difference_type operator-(const iterator& to,
                                                                    const iterator& from)
  {
    return to.map().distance(from.cursor(), to.cursor());
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator<(const iterator& left,
                                                         const iterator& right)
  {
    return left.map().less(left.cursor(), right.cursor());
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator>(const iterator& left,
                                                         const iterator& right)
  {
    return right.map().less(right.cursor(), left.cursor());
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator<=(const iterator& left,
                                                          const iterator& right)
  {
    return !(right.map().less(right.cursor(), left.cursor()));
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator>=(const iterator& left,
                                                          const iterator& right)
  {
    return !(left.map().less(left.cursor(), right.cursor()));
  }

 private:
  friend detail::mixin<Map, iterator<Map>>;

  [[nodiscard]] typename Map::cursor& cursor() noexcept { return cursor_map_pair_.first(); }

  [[nodiscard]] const typename Map::cursor& cursor() const noexcept
  {
    return cursor_map_pair_.first();
  }

  [[nodiscard]] Map& map() noexcept { return cursor_map_pair_.second(); }

  [[nodiscard]] const Map& map() const noexcept { return cursor_map_pair_.second(); }

  legate::detail::compressed_pair<typename Map::cursor, Map> cursor_map_pair_{};
};

template <typename Int>
class affine_map {
 public:
  using cursor = Int;

  template <typename Iterator>
  class mixin {
   public:
    [[nodiscard]] auto point() const
    {
      auto cursor        = static_cast<const Iterator&>(*this).cursor();
      auto shape         = static_cast<const Iterator&>(*this).map().shape();
      constexpr auto Dim = std::tuple_size_v<decltype(shape)>;
      Point<Dim> result;

      for (std::int32_t i = 0; i < Dim; ++i) {
        result[i] = cursor % shape[i];
        cursor /= shape[i];
      }
      return result;
    }
  };

  LEGATE_HOST_DEVICE [[nodiscard]] cursor next(cursor cur) const { return cur + 1; }

  LEGATE_HOST_DEVICE [[nodiscard]] cursor prev(cursor cur) const { return cur - 1; }

  LEGATE_HOST_DEVICE [[nodiscard]] cursor advance(cursor cur, std::ptrdiff_t n) const
  {
    return cur + n;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] std::ptrdiff_t distance(cursor from, cursor to) const
  {
    return to - from;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] bool less(cursor left, cursor right) const
  {
    return left < right;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] bool equal(cursor left, cursor right) const
  {
    return left == right;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] cursor begin() const { return 0; }
};

}  // namespace legate::stl

#include "suffix.hpp"
