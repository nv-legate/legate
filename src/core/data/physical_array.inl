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

// Useful for IDEs
#include "core/data/physical_array.h"

#include <utility>

namespace legate {

inline PhysicalArray::PhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl)
  : impl_{std::move(impl)}
{
}

inline const SharedPtr<detail::PhysicalArray>& PhysicalArray::impl() const { return impl_; }

template <int32_t DIM>
Rect<DIM> PhysicalArray::shape() const
{
  check_shape_dimension(DIM);
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
