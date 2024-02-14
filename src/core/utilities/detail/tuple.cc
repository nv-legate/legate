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

#include "core/utilities/detail/tuple.h"

namespace legate::detail {

Domain to_domain(const tuple<std::uint64_t>& shape)
{
  if (shape.empty()) {
    return {0, 0};
  }

  const auto ndim = static_cast<std::uint32_t>(shape.size());
  Domain domain;

  domain.dim = static_cast<int>(ndim);
  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    domain.rect_data[idx]        = 0;
    domain.rect_data[idx + ndim] = static_cast<coord_t>(shape[idx]) - 1;
  }
  return domain;
}

DomainPoint to_domain_point(const tuple<std::uint64_t>& shape)
{
  const auto ndim = static_cast<std::uint32_t>(shape.size());
  DomainPoint point;

  point.dim = static_cast<int>(ndim);
  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    point[idx] = static_cast<coord_t>(shape[idx]);
  }
  return point;
}

tuple<std::uint64_t> from_domain(const Domain& domain)
{
  std::vector<std::uint64_t> result;
  auto&& lo = domain.lo();
  auto&& hi = domain.hi();

  result.reserve(domain.dim);
  for (std::int32_t idx = 0; idx < domain.dim; ++idx) {
    result.emplace_back(hi[idx] - lo[idx] + 1);
  }
  return tuple<std::uint64_t>{std::move(result)};
}

}  // namespace legate::detail
