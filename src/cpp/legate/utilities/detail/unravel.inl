/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/unravel.h>

namespace legate::detail {

// This helper class converts indices to multi-dimensional points
template <std::int32_t NDIM>
LEGATE_HOST_DEVICE Unravel<NDIM>::Unravel(const Rect<NDIM>& rect) : low_{rect.lo}
{
  std::uint64_t stride = 1;
  for (std::int32_t dim = NDIM - 1; dim >= 0; --dim) {
    strides_[dim] = stride;
    stride *=
      static_cast<std::uint64_t>(std::max<std::int64_t>(rect.hi[dim] - rect.lo[dim] + 1, 0));
  }
  strides_[NDIM - 1] = stride;
}

template <std::int32_t NDIM>
LEGATE_HOST_DEVICE std::uint64_t Unravel<NDIM>::volume() const
{
  return strides_[NDIM - 1];
}

template <std::int32_t NDIM>
LEGATE_HOST_DEVICE bool Unravel<NDIM>::empty() const
{
  return volume() == 0;
}

template <std::int32_t NDIM>
LEGATE_HOST_DEVICE Point<NDIM> Unravel<NDIM>::operator()(std::uint64_t index) const
{
  auto point = low_;
  for (std::int32_t dim = 0; dim < NDIM - 1; dim++) {
    point[dim] += index / strides_[dim];
    index = index % strides_[dim];
  }
  point[NDIM - 1] += index;
  return point;
}

}  // namespace legate::detail
