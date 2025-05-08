/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace Realm {  // NOLINT

class ModuleConfig;

}  // namespace Realm

namespace legate::detail {

template <typename T>
class Argument;

void configure_cpus(bool auto_config,
                    const Realm::ModuleConfig& core,
                    const Argument<std::int32_t>& omps,
                    const Argument<std::int32_t>& util,
                    const Argument<std::int32_t>& gpus,
                    Argument<std::int32_t>* cpus);

}  // namespace legate::detail
