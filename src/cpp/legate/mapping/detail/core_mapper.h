/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>

#include <memory>

namespace legate::mapping::detail {

[[nodiscard]] std::unique_ptr<Mapper> create_core_mapper();

}  // namespace legate::mapping::detail
