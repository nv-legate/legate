/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/min_extents.h>

namespace legate::detail {

constexpr const ProxyMinExtents::value_type& ProxyMinExtents::variable() const noexcept
{
  return variable_;
}

constexpr const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& ProxyMinExtents::minimum_extents()
  const noexcept
{
  return minimum_extents_;
}

inline std::string_view ProxyMinExtents::name() const noexcept { return "min_extents"; }

}  // namespace legate::detail
