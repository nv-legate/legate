/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/device_mesh.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/hash.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <string>
#include <utility>

namespace legate::detail {

DeviceMesh::DeviceMesh(SmallVector<std::uint64_t, LEGATE_MAX_DIM> shape)
  : shape_{std::move(shape)}, shape_dp_{to_domain_point(shape_)}
{
  strides_dp_ = shape_dp_.strides();
  strides_.reserve(shape_.size());
  for (std::uint32_t dim = 0; dim < strides_dp_.dim(); ++dim) {
    strides_.push_back(static_cast<std::uint64_t>(strides_dp_[dim]));
  }
}

DeviceMesh::DeviceMesh(SmallVector<std::uint64_t, LEGATE_MAX_DIM> shape,
                       SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides)
  : shape_{std::move(shape)},
    strides_{std::move(strides)},
    shape_dp_{to_domain_point(shape_)},
    strides_dp_{to_domain_point(strides_)}
{
}

std::uint64_t DeviceMesh::num_cells() const
{
  return static_cast<std::uint64_t>(shape_dp_.volume());
}

std::uint64_t DeviceMesh::device_id(const DomainPoint& grid_point) const
{
  return static_cast<std::uint64_t>(strides_dp_.dot(grid_point));
}

DomainPoint DeviceMesh::grid_point(std::uint64_t device_id) const
{
  auto result = DomainPoint::zeroes(shape_dp_.dim());

  for (std::uint32_t dim = 0; dim < result.dim(); ++dim) {
    const auto stride = static_cast<std::uint64_t>(strides_dp_[dim]);
    const auto extent = static_cast<std::uint64_t>(shape_dp_[dim]);

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
