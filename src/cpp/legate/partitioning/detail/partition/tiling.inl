/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/tiling.h>

namespace legate::detail {

inline bool Tiling::is_convertible() const { return true; }

inline bool Tiling::is_invertible() const { return true; }

inline bool Tiling::has_launch_domain() const { return true; }

inline Span<const std::uint64_t> Tiling::tile_shape() const { return tile_shape_; }

inline bool Tiling::has_color_shape() const { return true; }

inline Span<const std::uint64_t> Tiling::color_shape() const { return color_shape_; }

inline Span<const std::int64_t> Tiling::offsets() const { return offsets_; }

inline Span<const std::uint64_t> Tiling::strides() const { return strides_; }

}  // namespace legate::detail
