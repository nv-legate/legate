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

namespace legate {

template <int32_t DIM>
Rect<DIM> Array::shape() const
{
  check_shape_dimension(DIM);
  if (dim() > 0) {
    return domain().bounds<DIM, coord_t>();
  } else {
    auto p = Point<DIM>::ZEROES();
    return Rect<DIM>(p, p);
  }
}

}  // namespace legate
