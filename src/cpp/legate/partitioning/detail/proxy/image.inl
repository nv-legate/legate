/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/image.h>

namespace legate::detail {

constexpr const ProxyImage::value_type& ProxyImage::var_function() const noexcept
{
  return var_function_;
}

constexpr const ProxyImage::value_type& ProxyImage::var_range() const noexcept
{
  return var_range_;
}

constexpr const std::optional<ImageComputationHint>& ProxyImage::hint() const noexcept
{
  return hint_;
}

inline std::string_view ProxyImage::name() const noexcept { return "image"; }

}  // namespace legate::detail
