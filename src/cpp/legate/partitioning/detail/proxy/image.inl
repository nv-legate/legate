/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
