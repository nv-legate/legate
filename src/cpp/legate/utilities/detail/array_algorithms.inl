/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/span.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>

namespace legate::detail {

template <typename T>
std::size_t array_volume(const T& container)
{
  static_assert(std::is_integral_v<typename T::value_type>);
  return std::reduce(
    std::begin(container), std::end(container), std::size_t{1}, std::multiplies<>{});
}

template <typename T>
T array_map(const T& container, Span<const std::int32_t> mapping)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    assert_valid_mapping(std::size(container), mapping);
  }

  T ret;

  ret.reserve(std::size(container));
  std::transform(mapping.begin(), mapping.end(), std::back_inserter(ret), [&](std::int32_t idx) {
    return container[idx];
  });
  return ret;
}

template <typename U, typename T>
U array_map(Span<const T> container, Span<const std::int32_t> mapping)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    assert_valid_mapping(container.size(), mapping);
  }

  U ret;

  ret.reserve(container.size());
  std::transform(mapping.begin(), mapping.end(), std::back_inserter(ret), [&](std::int32_t idx) {
    return container[idx];
  });
  return ret;
}

template <typename F, typename T, typename... Tn>
bool array_all_of(F&& func, const T& arr, const Tn&... rest)
{
  const auto zipper = detail::zip_equal(arr, rest...);

  return std::all_of(zipper.begin(), zipper.end(), [&](const auto& tup) {
    return std::apply(std::forward<F>(func), tup);
  });
}

}  // namespace legate::detail
