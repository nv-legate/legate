/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/store.h>

namespace legate::mapping {

template <std::int32_t DIM>
Rect<DIM> Store::shape() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  return Rect<DIM>{domain()};
}

inline const detail::Store* Store::impl() const noexcept { return impl_; }

inline Store::Store(const detail::Store* impl) noexcept : impl_{impl} {}

}  // namespace legate::mapping
