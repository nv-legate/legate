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

template <typename T>
class Scaled;

void configure_fbmem(bool auto_config,
                     const Realm::ModuleConfig* cuda,
                     const Argument<std::int32_t>& gpus,
                     Argument<Scaled<std::int64_t>>* fbmem);

}  // namespace legate::detail
