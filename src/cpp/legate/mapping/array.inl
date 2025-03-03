/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/array.h>

namespace legate::mapping {

template <std::int32_t DIM>
Rect<DIM> Array::shape() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  return Rect<DIM>{domain()};
}

inline Array::Array(const detail::Array* impl) : impl_{impl} {}

}  // namespace legate::mapping
