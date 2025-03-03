/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/mapper_manager.h>

namespace legate::detail {

inline Legion::MapperID MapperManager::mapper_id() const { return mapper_id_; }

}  // namespace legate::detail
