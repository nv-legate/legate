/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/align.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace legate::detail {

std::pair<void*, std::size_t> align_for_unpack_impl(void* ptr,
                                                    std::size_t capacity,
                                                    std::size_t bytes,
                                                    std::size_t align)
{
  // Ignoring any alignment attempts for zero-size items
  if (bytes == 0) {
    return {ptr, 0};
  }

  const auto orig_avail_space = std::min(bytes + align - 1, capacity);
  auto avail_space            = orig_avail_space;

  if (!std::align(align, bytes, ptr, avail_space)) {
    // If we get here, it means that someone did not pack the value correctly, likely without
    // first aligning the pointer!
    throw TracedException<std::runtime_error>{fmt::format(
      "Failed to align buffer {} (of size: {}) to {}-byte alignment (remaining capacity: {})",
      ptr,
      bytes,
      align,
      capacity)};
  }
  return {ptr, orig_avail_space - avail_space};
}

}  // namespace legate::detail
