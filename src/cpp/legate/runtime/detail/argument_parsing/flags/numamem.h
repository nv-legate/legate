/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

template <typename T>
class Argument;

template <typename T>
class Scaled;

void configure_numamem(bool auto_config,
                       Span<const std::size_t> numa_mems,
                       const Argument<std::int32_t>& omps,
                       Argument<Scaled<std::int64_t>>* numamem);

}  // namespace legate::detail
