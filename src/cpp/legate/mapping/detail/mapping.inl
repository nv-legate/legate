/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/mapping.h>

#include <utility>

namespace legate::mapping::detail {

inline DimOrdering::DimOrdering(Kind _kind) : kind{_kind} {}

inline DimOrdering::DimOrdering(std::vector<std::int32_t> _dims)
  : kind{Kind::CUSTOM}, dims{std::move(_dims)}
{
}

inline bool DimOrdering::operator==(const DimOrdering& other) const
{
  return kind == other.kind && dims == other.dims;
}

// ==========================================================================================

inline Span<const Store* const> StoreMapping::stores() const { return stores_; }

inline InstanceMappingPolicy& StoreMapping::policy() { return policy_; }

inline const InstanceMappingPolicy& StoreMapping::policy() const { return policy_; }

}  // namespace legate::mapping::detail
