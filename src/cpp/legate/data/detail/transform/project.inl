/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/project.h>

namespace legate::detail {

inline Project::Project(std::int32_t dim, std::int64_t coord) : dim_{dim}, coord_{coord} {}

inline std::int32_t Project::target_ndim(std::int32_t source_ndim) const { return source_ndim + 1; }

inline bool Project::is_convertible() const { return true; }

}  // namespace legate::detail
