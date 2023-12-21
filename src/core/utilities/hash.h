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

#include <functional>
#include <type_traits>
#include <utility>

namespace legate {

template <typename T, typename = void>
struct has_hash_member : std::false_type {};

template <typename T>
struct has_hash_member<
  T,
  std::void_t<std::enable_if_t<std::is_same_v<decltype(std::declval<T>().hash()), size_t>>>>
  : std::true_type {};

template <typename T>
constexpr bool has_hash_member_v = has_hash_member<T>::value;

template <typename T, typename = void>
struct hasher;

template <typename T>
struct hasher<T, std::enable_if_t<std::is_constructible_v<std::hash<T>>>> {
  [[nodiscard]] size_t operator()(const T& v) const noexcept { return std::hash<T>{}(v); }
};

template <typename T>
struct hasher<T, std::enable_if_t<!std::is_constructible_v<std::hash<T>> && has_hash_member_v<T>>> {
  [[nodiscard]] size_t operator()(const T& v) const noexcept { return v.hash(); }
};

inline void hash_combine(size_t&) noexcept {}

template <typename T, typename... Ts>
void hash_combine(size_t& target, const T& v, Ts&&... vs) noexcept
{
  // NOLINTNEXTLINE(readability-magic-numbers): the constants here are meant to be magic...
  target ^= hasher<T>{}(v) + 0x9e3779b9 + (target << 6) + (target >> 2);
  hash_combine(target, std::forward<Ts>(vs)...);
}

template <typename... Ts>
size_t hash_all(Ts&&... vs) noexcept
{
  size_t result = 0;
  hash_combine(result, std::forward<Ts>(vs)...);
  return result;
}

}  // namespace legate
