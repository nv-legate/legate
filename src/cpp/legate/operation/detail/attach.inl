/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/attach.h>

namespace legate::detail {

inline Operation::Kind Attach::kind() const { return Kind::ATTACH; }

inline bool Attach::needs_flush() const { return false; }

inline bool Attach::needs_partitioning() const { return false; }

}  // namespace legate::detail
