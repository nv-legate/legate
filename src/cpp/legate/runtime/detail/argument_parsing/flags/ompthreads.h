/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

class Config;

void configure_ompthreads(bool auto_config,
                          const Realm::ModuleConfig& core,
                          const Argument<std::int32_t>& util,
                          const Argument<std::int32_t>& cpus,
                          const Argument<std::int32_t>& gpus,
                          const Argument<std::int32_t>& omps,
                          Argument<std::int32_t>* ompthreads,
                          Config* cfg);

}  // namespace legate::detail
