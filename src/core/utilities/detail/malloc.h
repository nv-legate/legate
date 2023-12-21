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

#include <cassert>
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
    assert(num_elems >= 0);
  }
  assert(ret);
  *ret = static_cast<T*>(std::malloc(sizeof(T) * static_cast<size_t>(num_elems)));
}

}  // namespace legate::detail
