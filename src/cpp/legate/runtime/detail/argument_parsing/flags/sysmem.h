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

template <typename T>
class Scaled;

void configure_sysmem(bool auto_config,
                      const Realm::ModuleConfig& core,
                      const Argument<Scaled<std::int64_t>>& numamem,
                      Argument<Scaled<std::int64_t>>* sysmem);

}  // namespace legate::detail
