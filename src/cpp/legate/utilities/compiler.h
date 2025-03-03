/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <string>
#include <typeinfo>

namespace legate::detail {

[[nodiscard]] std::string demangle_type(const std::type_info&);

}  // namespace legate::detail
