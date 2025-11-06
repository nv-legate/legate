/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/transpose.h>

namespace legate::detail {

inline std::int32_t Transpose::target_ndim(std::int32_t source_ndim) const { return source_ndim; }

inline bool Transpose::is_convertible() const { return true; }

}  // namespace legate::detail
