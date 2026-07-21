/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/device_mesh.h>

namespace legate::detail {

inline const DomainPoint& DeviceMesh::shape() const { return shape_; }

inline const DomainPoint& DeviceMesh::strides() const { return strides_; }

inline std::uint32_t DeviceMesh::dim() const { return shape_.dim(); }

inline bool DeviceMesh::empty() const { return shape_.empty(); }

inline bool DeviceMesh::operator!=(const DeviceMesh& other) const { return !(*this == other); }

}  // namespace legate::detail
