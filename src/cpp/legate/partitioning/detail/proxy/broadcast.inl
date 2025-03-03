/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/broadcast.h>

namespace legate::detail {

constexpr const ProxyBroadcast::value_type& ProxyBroadcast::value() const noexcept
{
  return value_;
}

constexpr const std::optional<tuple<std::uint32_t>>& ProxyBroadcast::axes() const noexcept
{
  return axes_;
}

inline std::string_view ProxyBroadcast::name() const noexcept { return "broadcast"; }

}  // namespace legate::detail
