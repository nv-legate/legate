/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/index_attach.h>

namespace legate::detail {

inline Operation::Kind IndexAttach::kind() const { return Kind::INDEX_ATTACH; }

}  // namespace legate::detail
