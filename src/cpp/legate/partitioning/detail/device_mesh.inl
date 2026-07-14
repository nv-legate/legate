/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/device_mesh.h>

namespace legate::detail {

inline Span<const std::uint64_t> DeviceMesh::shape() const { return shape_; }

inline Span<const std::uint64_t> DeviceMesh::strides() const { return strides_; }

inline std::uint32_t DeviceMesh::dim() const { return static_cast<std::uint32_t>(shape_.size()); }

inline bool DeviceMesh::empty() const { return shape_.empty(); }

inline const DomainPoint& DeviceMesh::shape_dp() const { return shape_dp_; }

inline const DomainPoint& DeviceMesh::strides_dp() const { return strides_dp_; }

inline bool DeviceMesh::operator!=(const DeviceMesh& other) const { return !(*this == other); }

}  // namespace legate::detail
