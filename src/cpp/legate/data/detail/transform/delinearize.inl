/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/delinearize.h>

namespace legate::detail {

inline void Delinearize::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const {}

inline bool Delinearize::is_convertible() const { return false; }

}  // namespace legate::detail
