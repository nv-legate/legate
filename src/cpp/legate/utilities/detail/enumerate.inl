/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/enumerate.h>

#include <limits>

namespace legate::detail {

template <typename T>
constexpr CountingIterator<T>::CountingIterator(T v) noexcept : v_{v}
{
}

template <typename T>
constexpr typename CountingIterator<T>::value_type CountingIterator<T>::operator*() const noexcept
{
  return base();
}

template <typename T>
constexpr CountingIterator<T>& CountingIterator<T>::operator++() noexcept
{
  ++v_;
  return *this;
}

template <typename T>
constexpr CountingIterator<T> CountingIterator<T>::operator++(int) noexcept
{
  CountingIterator tmp = *this;

  ++v_;
  return tmp;
}

template <typename T>
constexpr CountingIterator<T>& CountingIterator<T>::operator--() noexcept
{
  --v_;
  return *this;
}

template <typename T>
constexpr CountingIterator<T> CountingIterator<T>::operator--(int) noexcept
{
  CountingIterator tmp = *this;

  --v_;
  return tmp;
}

template <typename T>
constexpr CountingIterator<T>& CountingIterator<T>::operator+=(difference_type n) noexcept
{
  v_ += static_cast<T>(n);
  return *this;
}

template <typename T>
constexpr CountingIterator<T>& CountingIterator<T>::operator-=(difference_type n) noexcept
{
  v_ -= static_cast<T>(n);
  return *this;
}

template <typename T>
constexpr CountingIterator<T> CountingIterator<T>::operator+(difference_type n) const noexcept
{
  return CountingIterator{base() + static_cast<T>(n)};
}

template <typename T>
constexpr CountingIterator<T> CountingIterator<T>::operator-(difference_type n) const noexcept
{
  return CountingIterator{base() - static_cast<T>(n)};
}

template <typename T>
constexpr typename CountingIterator<T>::difference_type CountingIterator<T>::operator-(
  const CountingIterator& other) const noexcept
{
  return static_cast<difference_type>(base() - other.base());
}

template <typename T>
constexpr typename CountingIterator<T>::value_type CountingIterator<T>::operator[](
  difference_type n) const noexcept
{
  return base() + static_cast<T>(n);
}

template <typename T>
constexpr bool CountingIterator<T>::operator==(const CountingIterator& o) const noexcept
{
  return base() == o.base();
}

template <typename T>
constexpr bool CountingIterator<T>::operator!=(const CountingIterator& o) const noexcept
{
  return base() != o.base();
}

template <typename T>
constexpr bool CountingIterator<T>::operator<(const CountingIterator& o) const noexcept
{
  return base() < o.base();
}

template <typename T>
constexpr bool CountingIterator<T>::operator>(const CountingIterator& o) const noexcept
{
  return base() > o.base();
}

template <typename T>
constexpr bool CountingIterator<T>::operator<=(const CountingIterator& o) const noexcept
{
  return base() <= o.base();
}

template <typename T>
constexpr bool CountingIterator<T>::operator>=(const CountingIterator& o) const noexcept
{
  return base() >= o.base();
}

template <typename T>
constexpr T CountingIterator<T>::base() const noexcept
{
  return v_;
}

// ==========================================================================================

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline constexpr Enumerator::Enumerator(value_type start) noexcept : start_{start} {}

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline constexpr typename Enumerator::value_type Enumerator::start() const noexcept
{
  return start_;
}

inline typename Enumerator::iterator Enumerator::begin() const noexcept
{
  return iterator{start()};
}

inline typename Enumerator::const_iterator Enumerator::cbegin() const noexcept
{
  return const_iterator{start()};
}

inline typename Enumerator::iterator Enumerator::end() const noexcept
{
  // An enumerator can never really be at the "end", so we just use the largest possible value
  // and hope that nobody ever gets that far.
  return iterator{std::numeric_limits<value_type>::max()};
}

inline typename Enumerator::const_iterator Enumerator::cend() const noexcept
{
  // An enumerator can never really be at the "end", so we just use the largest possible value
  // and hope that nobody ever gets that far.
  return const_iterator{std::numeric_limits<value_type>::max()};
}

// ==========================================================================================

template <typename T>
zip_detail::Zipper<zip_detail::ZiperatorShortest, Enumerator, T> enumerate(
  T&& iterable, typename Enumerator::value_type start)
{
  return zip_shortest(Enumerator{start}, std::forward<T>(iterable));
}

}  // namespace legate::detail
