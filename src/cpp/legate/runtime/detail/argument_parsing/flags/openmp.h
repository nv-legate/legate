/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <cstddef>
#include <cstdint>

namespace Realm {  // NOLINT

class ModuleConfig;

}  // namespace Realm

namespace legate::detail {

template <typename T>
class Argument;

void configure_omps(bool auto_config,
                    const Realm::ModuleConfig* openmp,
                    Span<const std::size_t> numa_mems,
                    const Argument<std::int32_t>& gpus,
                    Argument<std::int32_t>* omps);

}  // namespace legate::detail
