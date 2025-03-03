/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/scale.h>

namespace legate::detail {

constexpr const tuple<std::uint64_t>& ProxyScale::factors() const noexcept { return factors_; }

constexpr const ProxyScale::value_type& ProxyScale::var_smaller() const noexcept
{
  return var_smaller_;
}

constexpr const ProxyScale::value_type& ProxyScale::var_bigger() const noexcept
{
  return var_bigger_;
}

inline std::string_view ProxyScale::name() const noexcept { return "scale"; }

}  // namespace legate::detail
