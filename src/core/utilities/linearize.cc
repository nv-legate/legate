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

#include "core/utilities/linearize.h"
#include "core/utilities/dispatch.h"

namespace legate {

struct linearize_fn {
  template <int32_t DIM>
  size_t operator()(const DomainPoint& lo_dp, const DomainPoint& hi_dp, const DomainPoint& point_dp)
  {
    Point<DIM> lo      = lo_dp;
    Point<DIM> hi      = hi_dp;
    Point<DIM> point   = point_dp;
    Point<DIM> extents = hi - lo + Point<DIM>::ONES();
    size_t idx         = 0;
    for (int32_t dim = 0; dim < DIM; ++dim) idx = idx * extents[dim] + point[dim] - lo[dim];
    return idx;
  }
};

size_t linearize(const DomainPoint& lo, const DomainPoint& hi, const DomainPoint& point)
{
  return dim_dispatch(point.dim, linearize_fn{}, lo, hi, point);
}

struct delinearize_fn {
  template <int32_t DIM>
  DomainPoint operator()(const DomainPoint& lo_dp, const DomainPoint& hi_dp, size_t idx)
  {
    Point<DIM> lo      = lo_dp;
    Point<DIM> hi      = hi_dp;
    Point<DIM> extents = hi - lo + Point<DIM>::ONES();
    Point<DIM> point;
    for (int32_t dim = DIM - 1; dim >= 0; --dim) {
      point[dim] = idx % extents[dim] + lo[dim];
      idx /= extents[dim];
    }
    return point;
  }
};

DomainPoint delinearize(const DomainPoint& lo, const DomainPoint& hi, size_t idx)
{
  return dim_dispatch(lo.dim, delinearize_fn{}, lo, hi, idx);
}

}  // namespace legate
