/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/bloat.h>

namespace legate::detail {

constexpr const ProxyBloat::value_type& ProxyBloat::var_source() const noexcept
{
  return var_source_;
}

constexpr const ProxyBloat::value_type& ProxyBloat::var_bloat() const noexcept
{
  return var_bloat_;
}

constexpr const tuple<std::uint64_t>& ProxyBloat::low_offsets() const noexcept
{
  return low_offsets_;
}

constexpr const tuple<std::uint64_t>& ProxyBloat::high_offsets() const noexcept
{
  return high_offsets_;
}

inline std::string_view ProxyBloat::name() const noexcept { return "bloat"; }

}  // namespace legate::detail
