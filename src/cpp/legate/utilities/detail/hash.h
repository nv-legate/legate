/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/hash.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace legate {

template <typename T1, typename T2>
struct hasher<std::pair<T1, T2>> {
  [[nodiscard]] std::size_t operator()(const std::pair<T1, T2>& v) const noexcept
  {
    return hash_all(v.first, v.second);
  }
};

// Must be at least 1 item in the tuple
template <typename T, typename... U>
struct hasher<std::tuple<T, U...>> {
  [[nodiscard]] std::size_t operator()(const std::tuple<T, U...>& v) const noexcept
  {
    return std::apply(hash_all<T, U...>, v);
  }
};

}  // namespace legate

// NOLINTBEGIN(cert-dcl58-cpp, bugprone-std-namespace-modification)

template <>
struct std::hash<legate::Rect> {
  [[nodiscard]] std::size_t operator()(const legate::Rect& domain) const noexcept
  {
    std::size_t result = 0;
    auto&& lo          = domain.lo();
    auto&& hi          = domain.hi();

    for (std::uint32_t idx = 0; idx < domain.dim(); ++idx) {
      legate::hash_combine(result, lo[idx]);
      legate::hash_combine(result, hi[idx]);
    }
    return result;
  }
};

template <>
struct std::hash<legate::Point> {
  [[nodiscard]] std::size_t operator()(const legate::Point& domain_point) const noexcept
  {
    std::size_t result = 0;

    for (std::uint32_t idx = 0; idx < domain_point.dim(); ++idx) {
      legate::hash_combine(result, domain_point[idx]);
    }
    return result;
  }
};

template <typename T>
struct std::hash<std::reference_wrapper<T>> {
  [[nodiscard]] std::size_t operator()(const std::reference_wrapper<T>& v) const noexcept
  {
    return std::hash<std::decay_t<T>>{}(v.get());
  }
};

// NOLINTEND(cert-dcl58-cpp, bugprone-std-namespace-modification)
