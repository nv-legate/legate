/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/experimental/stl/detail/utility.hpp>
#include <legate/utilities/detail/compressed_pair.h>

#include <iterator>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

namespace detail {

template <typename Map>
using reference_t = decltype(std::declval<const Map&>().read(std::declval<typename Map::cursor>()));

template <typename Map, typename Iterator>
using mixin_ = typename Map::template mixin<Iterator>;

template <typename Map, typename Iterator>
using mixin = meta::eval<meta::quote_or<mixin_, meta::empty>, Map, Iterator>;

}  // namespace detail

template <typename Map>
class iterator  // NOLINT(readability-identifier-naming)
  : public detail::mixin<Map, iterator<Map>> {
 public:
  using difference_type   = std::ptrdiff_t;
  using value_type        = typename Map::value_type;
  using iterator_category = std::random_access_iterator_tag;
  using reference         = detail::reference_t<Map>;

  class pointer {  // NOLINT(readability-identifier-naming)
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
    cursor_() = map_().next(cursor_());
    return *this;
  }

  LEGATE_HOST_DEVICE iterator operator++(int)
  {
    auto copy = *this;
    ++*this;
    return copy;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] reference operator*() const { return map_().read(cursor_()); }

  LEGATE_HOST_DEVICE [[nodiscard]] pointer operator->() const { return pointer{operator*()}; }

  LEGATE_HOST_DEVICE friend bool operator==(const iterator& lhs, const iterator& rhs)
  {
    return lhs.map_().equal(lhs.cursor_(), rhs.cursor_());
  }

  LEGATE_HOST_DEVICE friend bool operator!=(const iterator& lhs, const iterator& rhs)
  {
    return !(lhs == rhs);
  }

  LEGATE_HOST_DEVICE iterator& operator--()
  {
    cursor_() = map_().prev(cursor_());
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
    return {map_(), map_().advance(cursor_(), n)};
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend iterator operator+(difference_type n, const iterator& it)
  {
    return it + n;
  }

  LEGATE_HOST_DEVICE iterator& operator+=(difference_type n)
  {
    cursor_() = map_().advance(cursor_(), n);
    return *this;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] iterator operator-(difference_type n) const
  {
    return {map_(), map_().advance(cursor_(), -n)};
  }

  LEGATE_HOST_DEVICE iterator& operator-=(difference_type n)
  {
    cursor_() = map_().advance(cursor_(), -n);
    return *this;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend difference_type operator-(const iterator& to,
                                                                    const iterator& from)
  {
    return to.map_().distance(from.cursor_(), to.cursor_());
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator<(const iterator& left,
                                                         const iterator& right)
  {
    return left.map_().less(left.cursor_(), right.cursor_());
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator>(const iterator& left,
                                                         const iterator& right)
  {
    return right.map_().less(right.cursor_(), left.cursor_());
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator<=(const iterator& left,
                                                          const iterator& right)
  {
    return !(right.map_().less(right.cursor_(), left.cursor_()));
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator>=(const iterator& left,
                                                          const iterator& right)
  {
    return !(left.map_().less(left.cursor_(), right.cursor_()));
  }

 private:
  friend detail::mixin<Map, iterator<Map>>;

  [[nodiscard]] typename Map::cursor& cursor_() noexcept { return cursor_map_pair_.first(); }

  [[nodiscard]] const typename Map::cursor& cursor_() const noexcept
  {
    return cursor_map_pair_.first();
  }

  [[nodiscard]] Map& map_() noexcept { return cursor_map_pair_.second(); }

  [[nodiscard]] const Map& map_() const noexcept { return cursor_map_pair_.second(); }

  legate::detail::CompressedPair<typename Map::cursor, Map> cursor_map_pair_{};
};

template <typename Int>
class affine_map {  // NOLINT(readability-identifier-naming)
 public:
  using cursor = Int;

  template <typename Iterator>
  class mixin {  // NOLINT(readability-identifier-naming)
   public:
    [[nodiscard]] auto point() const
    {
      auto cursor        = static_cast<const Iterator&>(*this).cursor();
      auto shape         = static_cast<const Iterator&>(*this).map().shape();
      constexpr auto DIM = std::tuple_size_v<decltype(shape)>;
      Point<DIM> result;

      for (std::int32_t i = 0; i < DIM; ++i) {
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

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
