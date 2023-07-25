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

#include "core/data/shape.h"

namespace legate {

Domain to_domain(const tuple<size_t>& shape)
{
  Domain domain;
  auto ndim  = static_cast<int32_t>(shape.size());
  domain.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) {
    domain.rect_data[idx]        = 0;
    domain.rect_data[idx + ndim] = static_cast<int64_t>(shape[idx]) - 1;
  }
  return domain;
}

DomainPoint to_domain_point(const Shape& shape)
{
  DomainPoint point;
  auto ndim = static_cast<int32_t>(shape.size());
  point.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) point[idx] = static_cast<int64_t>(shape[idx]);
  return point;
}

Shape from_domain(const Domain& domain)
{
  std::vector<size_t> result;
  auto lo = domain.lo();
  auto hi = domain.hi();
  for (int32_t idx = 0; idx < domain.dim; ++idx) result.push_back(hi[idx] - lo[idx] + 1);
  return Shape(std::move(result));
}

}  // namespace legate
