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

void configure_gpus(bool auto_config,
                    const Realm::ModuleConfig* cuda,
                    Argument<std::int32_t>* gpus,
                    Config* cfg);

}  // namespace legate::detail
