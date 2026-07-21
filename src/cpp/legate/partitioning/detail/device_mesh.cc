/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/device_mesh.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/hash.h>

#include <fmt/format.h>

#include <cstdint>
#include <string>
#include <utility>

namespace legate::detail {

DeviceMesh::DeviceMesh(DomainPoint shape) : shape_{std::move(shape)}, strides_{shape_.strides()} {}

DeviceMesh::DeviceMesh(DomainPoint shape, DomainPoint strides)
  : shape_{std::move(shape)}, strides_{std::move(strides)}
{
}

std::uint64_t DeviceMesh::num_cells() const { return static_cast<std::uint64_t>(shape_.volume()); }

std::uint64_t DeviceMesh::device_id(const DomainPoint& grid_point) const
{
  return static_cast<std::uint64_t>(strides_.dot(grid_point));
}

DomainPoint DeviceMesh::grid_point(std::uint64_t device_id) const
{
  auto result = DomainPoint::zeroes(static_cast<std::int32_t>(shape_.dim()));

  for (std::uint32_t dim = 0; dim < result.dim(); ++dim) {
    const auto stride = static_cast<std::uint64_t>(strides_[dim]);
    const auto extent = static_cast<std::uint64_t>(shape_[dim]);

    result[dim] = stride == 0 ? 0 : static_cast<coord_t>((device_id / stride) % extent);
  }

  LEGATE_ASSERT(this->device_id(result) == device_id);
  return result;
}

bool DeviceMesh::operator==(const DeviceMesh& other) const
{
  return shape_ == other.shape_ && strides_ == other.strides_;
}

std::size_t DeviceMesh::hash() const { return hash_all(shape_, strides_); }

std::string DeviceMesh::to_string() const
{
  return fmt::format("DeviceMesh(shape: {}, strides: {})", shape_, strides_);
}

}  // namespace legate::detail
