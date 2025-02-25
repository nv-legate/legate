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

#include <legate/partitioning/detail/proxy/bloat.h>

namespace legate::detail::proxy {

constexpr const Bloat::value_type& Bloat::var_source() const noexcept { return var_source_; }

constexpr const Bloat::value_type& Bloat::var_bloat() const noexcept { return var_bloat_; }

constexpr const tuple<std::uint64_t>& Bloat::low_offsets() const noexcept { return low_offsets_; }

constexpr const tuple<std::uint64_t>& Bloat::high_offsets() const noexcept { return high_offsets_; }

inline std::string_view Bloat::name() const noexcept { return "bloat"; }

}  // namespace legate::detail::proxy
