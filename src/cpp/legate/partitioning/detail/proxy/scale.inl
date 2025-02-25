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

#include <legate/partitioning/detail/proxy/scale.h>

namespace legate::detail::proxy {

constexpr const tuple<std::uint64_t>& Scale::factors() const noexcept { return factors_; }

constexpr const Scale::value_type& Scale::var_smaller() const noexcept { return var_smaller_; }

constexpr const Scale::value_type& Scale::var_bigger() const noexcept { return var_bigger_; }

inline std::string_view Scale::name() const noexcept { return "scale"; }

}  // namespace legate::detail::proxy
