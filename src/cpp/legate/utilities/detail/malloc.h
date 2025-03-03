/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/assert.h>

#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace legate::detail {

// call it typed_malloc so it does not clash with global malloc()
template <typename T, typename U>
void typed_malloc(T** ret, U num_elems) noexcept
{
  static_assert(std::is_integral_v<U>);
  if constexpr (std::is_signed_v<U>) {
    LEGATE_CHECK(num_elems >= 0);
  }
  LEGATE_CHECK(ret);
  *ret = static_cast<T*>(std::malloc(sizeof(T) * static_cast<std::size_t>(num_elems)));
}

}  // namespace legate::detail
