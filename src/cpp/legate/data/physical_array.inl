/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/data/physical_array.h>

namespace legate {

inline const SharedPtr<detail::PhysicalArray>& PhysicalArray::impl() const { return impl_; }

template <std::int32_t DIM>
Rect<DIM> PhysicalArray::shape() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_shape_dimension_(DIM);
  if (dim() > 0) {
    return domain().bounds<DIM, coord_t>();
  }

  static const auto ret = Rect<DIM>{Point<DIM>::ZEROES(), Point<DIM>::ZEROES()};

  return ret;
}

inline const std::optional<LogicalArray>& PhysicalArray::owner() const { return owner_; }

}  // namespace legate
