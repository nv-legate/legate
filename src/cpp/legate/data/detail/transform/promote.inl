/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/promote.h>

namespace legate::detail {

inline Promote::Promote(std::int32_t extra_dim, std::int64_t dim_size)
  : extra_dim_{extra_dim}, dim_size_{dim_size}
{
}

inline std::int32_t Promote::target_ndim(std::int32_t source_ndim) const { return source_ndim - 1; }

inline bool Promote::is_convertible() const { return true; }

}  // namespace legate::detail
