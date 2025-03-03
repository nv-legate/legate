/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace legate {

template <typename T, typename = void>
struct has_hash_member : std::false_type {};

template <typename T>
struct has_hash_member<
  T,
  std::void_t<std::enable_if_t<std::is_same_v<decltype(std::declval<T>().hash()), std::size_t>>>>
  : std::true_type {};

template <typename T>
inline constexpr bool has_hash_member_v = has_hash_member<T>::value;

template <typename T, typename = void>
struct hasher;

template <typename T>
struct hasher<T, std::enable_if_t<std::is_constructible_v<std::hash<T>>>> {
  [[nodiscard]] constexpr std::size_t operator()(const T& v) const noexcept
  {
    return std::hash<T>{}(v);
  }
};

template <typename T>
struct hasher<T, std::enable_if_t<!std::is_constructible_v<std::hash<T>> && has_hash_member_v<T>>> {
  [[nodiscard]] constexpr std::size_t operator()(const T& v) const noexcept { return v.hash(); }
};

template <typename T>
constexpr void hash_combine(std::size_t& target, const T& v) noexcept
{
  // NOLINTNEXTLINE(readability-magic-numbers): the constants here are meant to be magic...
  target ^= hasher<T>{}(v) + 0x9e3779b9 + (target << 6) + (target >> 2);
}

template <typename... T>
[[nodiscard]] constexpr std::size_t hash_all(const T&... vs) noexcept
{
  std::size_t result = 0;

  (hash_combine(result, vs), ...);
  return result;
}

}  // namespace legate
