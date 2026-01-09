/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/compiler.h>

#include <fmt/std.h>

#include <string>
#include <typeinfo>

namespace legate::detail {

std::string demangle_type(const std::type_info& ti) { return fmt::format("{}", ti); }

}  // namespace legate::detail
