/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace legate::detail {

enum class ExceptionKind : std::uint8_t { CPP, PYTHON };

}  // namespace legate::detail
