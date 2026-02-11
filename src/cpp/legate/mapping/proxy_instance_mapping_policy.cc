/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/proxy_instance_mapping_policy.h>

#include <tuple>

namespace legate::mapping {

bool operator==(const ProxyInstanceMappingPolicy& lhs, const ProxyInstanceMappingPolicy& rhs)
{
  // Do ordering comparison last, it's the most expensive
  return std::tie(lhs.target, lhs.allocation, lhs.exact, lhs.redundant, lhs.ordering) ==
         std::tie(rhs.target, rhs.allocation, rhs.exact, rhs.redundant, rhs.ordering);
}

bool operator!=(const ProxyInstanceMappingPolicy& lhs, const ProxyInstanceMappingPolicy& rhs)
{
  return !(lhs == rhs);
}

}  // namespace legate::mapping
