/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/tiling.h>

namespace legate::detail {

inline bool Tiling::is_convertible() const { return true; }

inline bool Tiling::is_invertible() const { return true; }

inline const DomainPoint& Tiling::tile_shape() const { return tile_shape_; }

inline bool Tiling::has_color_shape() const { return true; }

inline const DomainPoint& Tiling::color_shape() const { return color_shape_; }

inline const DomainPoint& Tiling::offsets() const { return offsets_; }

inline const DomainPoint& Tiling::strides() const { return strides_; }

inline const DeviceMesh& Tiling::device_mesh() const { return device_mesh_; }

}  // namespace legate::detail
