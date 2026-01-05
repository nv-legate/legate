/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/align.h>

#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

namespace legate::detail {

template <typename T>
[[nodiscard]] std::pair<void*, std::size_t> pack_buffer(void* buf,
                                                        std::size_t remaining_cap,
                                                        T&& value)
{
  const auto [ptr, align_offset] = align_for_unpack<T>(buf, remaining_cap);

  *static_cast<std::decay_t<T>*>(ptr) = std::forward<T>(value);
  return {static_cast<char*>(ptr) + sizeof(T), remaining_cap - sizeof(T) - align_offset};
}

template <typename T>
[[nodiscard]] std::pair<void*, std::size_t> pack_buffer(void* buf,
                                                        std::size_t remaining_cap,
                                                        std::size_t nelem,
                                                        const T* value)
{
  const auto copy_bytes = nelem * sizeof(T);

  if (!copy_bytes) {
    return {buf, remaining_cap};
  }

  const auto [ptr, align_offset] = align_for_unpack<T>(buf, remaining_cap);

  LEGATE_ASSERT(value);
  std::memcpy(ptr, value, copy_bytes);
  return {static_cast<char*>(ptr) + copy_bytes, remaining_cap - copy_bytes - align_offset};
}

template <typename T>
[[nodiscard]] std::pair<const void*, std::size_t> unpack_buffer(const void* buf,
                                                                std::size_t remaining_cap,
                                                                T* value)
{
  const auto [ptr, align_offset] = align_for_unpack<T>(const_cast<void*>(buf), remaining_cap);

  *value = *static_cast<std::decay_t<T>*>(ptr);
  return {static_cast<char*>(ptr) + sizeof(T), remaining_cap - sizeof(T) - align_offset};
}

template <typename T>
[[nodiscard]] std::pair<const void*, std::size_t> unpack_buffer(const void* buf,
                                                                std::size_t remaining_cap,
                                                                std::size_t nelem,
                                                                T* const* value)
{
  const auto copy_bytes = nelem * sizeof(T);

  if (!copy_bytes) {
    return {buf, remaining_cap};
  }

  const auto [ptr, align_offset] = align_for_unpack<T>(const_cast<void*>(buf), remaining_cap);

  LEGATE_ASSERT(*value);
  std::memcpy(*value, ptr, copy_bytes);
  return {static_cast<char*>(ptr) + copy_bytes, remaining_cap - copy_bytes - align_offset};
}

}  // namespace legate::detail
