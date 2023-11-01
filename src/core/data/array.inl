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
#include "core/data/array.h"

#include <utility>

namespace legate {

inline Array::Array(std::shared_ptr<detail::Array> impl) : impl_{std::move(impl)} {}

inline const std::shared_ptr<detail::Array>& Array::impl() const { return impl_; }

template <int32_t DIM>
Rect<DIM> Array::shape() const
{
  check_shape_dimension(DIM);
  if (dim() > 0) return domain().bounds<DIM, coord_t>();
  auto p = Point<DIM>::ZEROES();
  return {p, p};
}

// ==========================================================================================

inline ListArray::ListArray(std::shared_ptr<detail::Array> impl) : Array{std::move(impl)} {}

// ==========================================================================================

inline StringArray::StringArray(std::shared_ptr<detail::Array> impl) : Array{std::move(impl)} {}

}  // namespace legate
