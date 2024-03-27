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

#include "core/utilities/deserializer.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace legate::detail {

inline namespace ret_exn_util {

template <typename T>
[[nodiscard]] std::pair<void*, std::size_t> pack_buffer(void* buf,
                                                        std::size_t remaining_cap,
                                                        T&& value)
{
  const auto [ptr, align_offset] = detail::align_for_unpack<T>(buf, remaining_cap);

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

  const auto [ptr, align_offset] = detail::align_for_unpack<T>(buf, remaining_cap);

  LegateAssert(value);
  std::memcpy(ptr, value, copy_bytes);
  return {static_cast<char*>(ptr) + copy_bytes, remaining_cap - copy_bytes - align_offset};
}

template <typename T>
[[nodiscard]] std::pair<const void*, std::size_t> unpack_buffer(const void* buf,
                                                                std::size_t remaining_cap,
                                                                T* value)
{
  const auto [ptr, align_offset] =
    legate::detail::align_for_unpack<T>(const_cast<void*>(buf), remaining_cap);

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

  const auto val_ptr = *value;
  const auto [ptr, align_offset] =
    legate::detail::align_for_unpack<T>(const_cast<void*>(buf), remaining_cap);

  LegateAssert(val_ptr);
  std::memcpy(val_ptr, ptr, copy_bytes);
  return {static_cast<char*>(val_ptr) + copy_bytes, remaining_cap - copy_bytes - align_offset};
}

template <typename T>
constexpr std::size_t max_aligned_size_for_type()
{
  static_assert(!std::is_reference_v<T>);
  // NOLINTNEXTLINE(bugprone-sizeof-expression) // comparison to 0 is the point
  static_assert(sizeof(T) > 0, "Cannot be used for incomplete type");
  return sizeof(T) + alignof(T) - 1;
}

}  // namespace ret_exn_util

enum class ExceptionKind : std::uint8_t { CPP, PYTHON };

}  // namespace legate::detail
