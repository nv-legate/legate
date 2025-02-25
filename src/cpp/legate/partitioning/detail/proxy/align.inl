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

#include <legate/partitioning/detail/proxy/align.h>

namespace legate::detail::proxy {

constexpr const Align::value_type& Align::left() const noexcept { return left_; }

constexpr const Align::value_type& Align::right() const noexcept { return right_; }

inline std::string_view Align::name() const noexcept { return "align"; }

}  // namespace legate::detail::proxy
