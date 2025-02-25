/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/partitioning/detail/proxy/broadcast.h>

namespace legate::detail::proxy {

constexpr const Broadcast::value_type& Broadcast::value() const noexcept { return value_; }

constexpr const std::optional<tuple<std::uint32_t>>& Broadcast::axes() const noexcept
{
  return axes_;
}

inline std::string_view Broadcast::name() const noexcept { return "broadcast"; }

}  // namespace legate::detail::proxy
