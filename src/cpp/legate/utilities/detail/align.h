/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace legate::detail {

[[nodiscard]] constexpr std::size_t round_up_to_multiple(std::size_t value, std::size_t round_to)
{
  return (value + round_to - 1) / round_to * round_to;
}

[[nodiscard]] std::pair<void*, std::size_t> align_for_unpack_impl(void* ptr,
                                                                  std::size_t capacity,
                                                                  std::size_t bytes,
                                                                  std::size_t align);

template <typename T>
[[nodiscard]] std::pair<void*, std::size_t> align_for_unpack(
  void* ptr,
  std::size_t capacity,
  // It's OK if T is a pointer. We trust that the caller knows what they are doing
  std::size_t bytes = sizeof(T),  // NOLINT(bugprone-sizeof-expression)
  std::size_t align = alignof(T))
{
  return align_for_unpack_impl(ptr, capacity, bytes, align);
}

template <typename T>
constexpr std::size_t max_aligned_size_for_type()
{
  static_assert(!std::is_reference_v<T>);
  // NOLINTNEXTLINE(bugprone-sizeof-expression) // comparison to 0 is the point
  static_assert(sizeof(T) > 0, "Cannot be used for incomplete type");
  return sizeof(T) + alignof(T) - 1;
}

}  // namespace legate::detail
