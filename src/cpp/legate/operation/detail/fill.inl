/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/fill.h>

namespace legate::detail {

inline Operation::Kind Fill::kind() const { return Kind::FILL; }

}  // namespace legate::detail
