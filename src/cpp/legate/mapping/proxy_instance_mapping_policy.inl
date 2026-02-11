/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/proxy_instance_mapping_policy.h>

#include <utility>

namespace legate::mapping {

inline ProxyInstanceMappingPolicy&& ProxyInstanceMappingPolicy::with_target(
  std::optional<StoreTarget> tgt) &&
{
  target = tgt;
  return std::move(*this);
}

inline ProxyInstanceMappingPolicy&& ProxyInstanceMappingPolicy::with_allocation_policy(
  AllocPolicy alloc) &&
{
  allocation = alloc;
  return std::move(*this);
}

inline ProxyInstanceMappingPolicy&& ProxyInstanceMappingPolicy::with_ordering(
  std::optional<DimOrdering> ord) &&
{
  ordering = std::move(ord);
  return std::move(*this);
}

inline ProxyInstanceMappingPolicy&& ProxyInstanceMappingPolicy::with_exact(bool value) &&
{
  exact = value;
  return std::move(*this);
}

inline ProxyInstanceMappingPolicy&& ProxyInstanceMappingPolicy::with_redundant(bool value) &&
{
  redundant = value;
  return std::move(*this);
}

}  // namespace legate::mapping
