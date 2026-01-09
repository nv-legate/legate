/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/cpp_version.h>
#include <legate/utilities/detail/zip.h>

#include <cstddef>
#include <iterator>
#include <type_traits>

LEGATE_CPP_VERSION_TODO(23,
                        "Can remove this module in favor of std::ranges::views::enumerate and/or "
                        "std::ranges::enumerate_view");

namespace legate::detail {

template <typename T>
class CountingIterator {
 public:
  using value_type        = std::remove_cv_t<T>;
  using reference         = value_type;
  using pointer           = void;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;

  constexpr CountingIterator() noexcept = default;
  explicit constexpr CountingIterator(T v) noexcept;

  [[nodiscard]] constexpr value_type operator*() const noexcept;

  constexpr CountingIterator& operator++() noexcept;
  constexpr CountingIterator operator++(int) noexcept;

  constexpr CountingIterator& operator--() noexcept;
  constexpr CountingIterator operator--(int) noexcept;

  constexpr CountingIterator& operator+=(difference_type n) noexcept;
  constexpr CountingIterator& operator-=(difference_type n) noexcept;
  constexpr CountingIterator operator+(difference_type n) const noexcept;
  constexpr CountingIterator operator-(difference_type n) const noexcept;

  friend constexpr CountingIterator operator+(difference_type n, CountingIterator it) noexcept
  {
    return it + n;
  }

  [[nodiscard]] constexpr difference_type operator-(const CountingIterator& other) const noexcept;
  [[nodiscard]] constexpr value_type operator[](difference_type n) const noexcept;

  [[nodiscard]] constexpr bool operator==(const CountingIterator& o) const noexcept;
  [[nodiscard]] constexpr bool operator!=(const CountingIterator& o) const noexcept;
  [[nodiscard]] constexpr bool operator<(const CountingIterator& o) const noexcept;
  [[nodiscard]] constexpr bool operator>(const CountingIterator& o) const noexcept;
  [[nodiscard]] constexpr bool operator<=(const CountingIterator& o) const noexcept;
  [[nodiscard]] constexpr bool operator>=(const CountingIterator& o) const noexcept;

  [[nodiscard]] constexpr T base() const noexcept;

 private:
  T v_{};
};

// ==========================================================================================

class Enumerator {
 public:
  using iterator          = CountingIterator<std::ptrdiff_t>;
  using const_iterator    = CountingIterator<std::ptrdiff_t>;
  using value_type        = typename iterator::value_type;
  using iterator_category = typename iterator::iterator_category;
  using difference_type   = typename iterator::difference_type;
  using pointer           = typename iterator::pointer;
  using reference         = typename iterator::reference;

  constexpr Enumerator() noexcept = default;
  constexpr explicit Enumerator(value_type start) noexcept;

  [[nodiscard]] constexpr value_type start() const noexcept;

  [[nodiscard]] iterator begin() const noexcept;
  [[nodiscard]] const_iterator cbegin() const noexcept;

  [[nodiscard]] iterator end() const noexcept;
  [[nodiscard]] const_iterator cend() const noexcept;

 private:
  value_type start_{};
};

/**
 * @brief Enumerate an iterable
 *
 * @param iterable The iterable to enumerate
 * @param start [optional] Set the starting value for the enumerator
 *
 * @return The enumerator iterator adaptor
 *
 * @details The enumerator is classed as a bidirectional iterator, so can be both incremented
 * and decremented. Decrementing the enumerator will decrease the count. However, this only
 * applies if \p iterable is itself at least bidirectional. If \p iterable does not satisfy
 * bidirectional iteration, then the returned enumerator will assume the iterator category of
 * \p iterable.
 *
 * @snippet noinit/enumerate.cc Constructing an enumerator
 */
template <typename T>
[[nodiscard]] zip_detail::Zipper<zip_detail::ZiperatorShortest, Enumerator, T> enumerate(
  T&& iterable, typename Enumerator::value_type start = {});

}  // namespace legate::detail

#include <legate/utilities/detail/enumerate.inl>
