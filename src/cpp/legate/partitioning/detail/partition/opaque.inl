/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/opaque.h>

namespace legate::detail {

inline Opaque::Opaque(DomainPoint color_shape) : color_shape_{std::move(color_shape)} {}

inline bool Opaque::is_convertible() const { return false; }

inline bool Opaque::is_invertible() const { return false; }

inline bool Opaque::has_color_shape() const { return !color_shape_.empty(); }

inline const DomainPoint& Opaque::color_shape() const { return color_shape_; }

inline bool Opaque::is_equi_size(const Domain&) const { return false; }

inline bool Opaque::filled() const { return filled_; }

}  // namespace legate::detail
