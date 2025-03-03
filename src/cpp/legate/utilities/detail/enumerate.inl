/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/enumerate.h>

#include <limits>

namespace legate::detail {

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
