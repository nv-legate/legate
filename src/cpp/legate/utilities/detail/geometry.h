/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

[[nodiscard]] std::int64_t inner_product(const DomainPoint& rhs1, const DomainPoint& rhs2) noexcept;

[[nodiscard]] std::int64_t calculate_volume(const DomainPoint& shape) noexcept;

[[nodiscard]] DomainPoint calculate_strides(const Domain& domain) noexcept;

[[nodiscard]] DomainPoint calculate_strides(const DomainPoint& shape) noexcept;

[[nodiscard]] DomainPoint to_shape(const Domain& domain) noexcept;

}  // namespace legate::detail
