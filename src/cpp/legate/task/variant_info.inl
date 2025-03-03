/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/variant_info.h>

namespace legate {

inline const detail::VariantInfo& VariantInfo::impl_() const noexcept { return *pimpl_; }

inline VariantInfo::VariantInfo(const detail::VariantInfo& impl) noexcept : pimpl_{&impl} {}

}  // namespace legate
