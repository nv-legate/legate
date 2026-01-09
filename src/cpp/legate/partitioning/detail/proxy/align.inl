/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/align.h>

namespace legate::detail {

constexpr const ProxyAlign::value_type& ProxyAlign::left() const noexcept { return left_; }

constexpr const ProxyAlign::value_type& ProxyAlign::right() const noexcept { return right_; }

inline std::string_view ProxyAlign::name() const noexcept { return "align"; }

}  // namespace legate::detail
