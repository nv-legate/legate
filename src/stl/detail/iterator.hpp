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

  LEGATE_STL_ATTRIBUTE((host, device))            //
  iterator(Map map, typename Map::cursor cursor)  //
    : cursor_{cursor}, map_{std::move(map)}
  {
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator& operator++()
  {
    cursor_ = map_.next(cursor_);
    return *this;
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator operator++(int)
  {
    auto copy = *this;
    ++*this;
    return copy;
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] reference operator*() const { return map_.read(cursor_); }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] pointer operator->() const { return pointer{operator*()}; }

  LEGATE_STL_ATTRIBUTE((host, device))
  friend bool operator==(const iterator& lhs, const iterator& rhs)
  {
    return lhs.map_.equal(lhs.cursor_, rhs.cursor_);
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  friend bool operator!=(const iterator& lhs, const iterator& rhs) { return !(lhs == rhs); }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator& operator--()
  {
    cursor_ = map_.prev(cursor_);
    return *this;
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator operator--(int)
  {
    auto copy = *this;
    --*this;
    return copy;
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] iterator operator+(difference_type n) const
  {
    return {map_, map_.advance(cursor_, n)};
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  [[nodiscard]] friend iterator operator+(difference_type n, const iterator& it) { return it + n; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator& operator+=(difference_type n)
  {
    cursor_ = map_.advance(cursor_, n);
    return *this;
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] iterator operator-(difference_type n) const
  {
    return {map_, map_.advance(cursor_, -n)};
  }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator& operator-=(difference_type n)
  {
    cursor_ = map_.advance(cursor_, -n);
    return *this;
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  [[nodiscard]] friend difference_type operator-(const iterator& to, const iterator& from)
  {
    return to.map_.distance(from.cursor_, to.cursor_);
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  [[nodiscard]] friend bool operator<(const iterator& left, const iterator& right)
  {
    return left.map_.less(left.cursor_, right.cursor_);
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  [[nodiscard]] friend bool operator>(const iterator& left, const iterator& right)
  {
    return right.map_.less(right.cursor_, left.cursor_);
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  [[nodiscard]] friend bool operator<=(const iterator& left, const iterator& right)
  {
    return !(right.map_.less(right.cursor_, left.cursor_));
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  [[nodiscard]] friend bool operator>=(const iterator& left, const iterator& right)
  {
    return !(left.map_.less(left.cursor_, right.cursor_));
  }

 private:
  friend detail::mixin<Map, iterator<Map>>;

  [[nodiscard]] typename Map::cursor cursor() const { return cursor_; }

  [[nodiscard]] Map& map() { return map_; }

  [[nodiscard]] const Map& map() const { return map_; }

  typename Map::cursor cursor_{};
  LEGATE_STL_ATTRIBUTE((no_unique_address)) Map map_{};
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

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] cursor next(cursor cur) const { return cur + 1; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] cursor prev(cursor cur) const { return cur - 1; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] cursor advance(cursor cur, std::ptrdiff_t n) const { return cur + n; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] std::ptrdiff_t distance(cursor from, cursor to) const { return to - from; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] bool less(cursor left, cursor right) const { return left < right; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] bool equal(cursor left, cursor right) const { return left == right; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] cursor begin() const { return 0; }
};

}  // namespace legate::stl

#include "suffix.hpp"
