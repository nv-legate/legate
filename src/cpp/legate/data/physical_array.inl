/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/data/physical_array.h>

#include <utility>

namespace legate {

inline PhysicalArray::PhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl)
  : impl_{std::move(impl)}
{
}

inline const SharedPtr<detail::PhysicalArray>& PhysicalArray::impl() const { return impl_; }

template <std::int32_t DIM>
Rect<DIM> PhysicalArray::shape() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_shape_dimension_(DIM);
  if (dim() > 0) {
    return domain().bounds<DIM, coord_t>();
  }
  auto p = Point<DIM>::ZEROES();
  return {p, p};
}

// ==========================================================================================

inline ListPhysicalArray::ListPhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl)
  : PhysicalArray{std::move(impl)}
{
}

// ==========================================================================================

inline StringPhysicalArray::StringPhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl)
  : PhysicalArray{std::move(impl)}
{
}

}  // namespace legate
