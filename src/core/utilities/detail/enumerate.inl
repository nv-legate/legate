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

#include "core/utilities/detail/enumerate.h"

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
  return zip(Enumerator{start}, std::forward<T>(iterable));
}

}  // namespace legate::detail
