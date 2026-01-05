/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/type/half.h>

#pragma once

namespace legate {

#if LEGATE_DEFINED(LEGATE_DEFINED_HALF)
constexpr Half::Half(std::uint16_t a) noexcept : repr_{a} {}

constexpr std::uint16_t Half::raw() const noexcept { return repr_; }

inline bool operator==(const Half& a, const Half& b) noexcept { return a.raw() == b.raw(); }

inline bool operator!=(const Half& a, const Half& b) noexcept { return !(a == b); }
#endif

}  // namespace legate
