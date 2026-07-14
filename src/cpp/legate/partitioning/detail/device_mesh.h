/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace legate::detail {

/**
 * @brief A multi-dimensional grid view of the devices participating in a partition.
 *
 * A `DeviceMesh` decouples the "colors" of a partition (the names of tiles) from the physical
 * devices that hold them. It has two pieces of state:
 *
 *  - a `shape`, describing an N-dimensional grid of devices (e.g. a shape of `(2, 3)` lays the
 *    devices out in a 2x3 grid); and
 *  - a tuple of `strides`, describing how a grid coordinate is linearized into a device id. For
 *    example, a mesh of shape `(2, 3)` with strides `(3, 1)` maps grid points `(0, 1)` and
 *    `(1, 1)` to device ids `1` and `4`; with strides `(1, 2)` the same grid points map to device
 *    ids `2` and `3`.
 *
 * Because a partition's color shape need not use every dimension of the mesh, a `DeviceMesh`
 * dimension that is not driven by a color dimension becomes a replication axis: a single tile is
 * mapped to every device along that axis. This is how partial replication is expressed (see
 * `Tiling`).
 */
class DeviceMesh {
 public:
  DeviceMesh() = default;

  /**
   * @brief Construct a mesh from a shape, using row-major (C-order) strides.
   *
   * @param shape The extents of the device grid.
   */
  explicit DeviceMesh(SmallVector<std::uint64_t, LEGATE_MAX_DIM> shape);

  /**
   * @brief Construct a mesh from an explicit shape and strides.
   *
   * @param shape   The extents of the device grid.
   * @param strides The per-dimension strides used to linearize a grid coordinate into a device id.
   */
  DeviceMesh(SmallVector<std::uint64_t, LEGATE_MAX_DIM> shape,
             SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides);

  [[nodiscard]] Span<const std::uint64_t> shape() const;
  [[nodiscard]] Span<const std::uint64_t> strides() const;
  [[nodiscard]] std::uint32_t dim() const;
  [[nodiscard]] bool empty() const;

  [[nodiscard]] const DomainPoint& shape_dp() const;
  [[nodiscard]] const DomainPoint& strides_dp() const;

  /**
   * @brief The number of grid cells in the mesh (the product of its shape).
   */
  [[nodiscard]] std::uint64_t num_cells() const;

  /**
   * @brief Linearize a grid coordinate into a device id.
   *
   * @param grid_point A coordinate in the device grid (must have `dim()` dimensions).
   *
   * @return The device id `sum_d(strides[d] * grid_point[d])`.
   */
  [[nodiscard]] std::uint64_t device_id(const DomainPoint& grid_point) const;

  /**
   * @brief Invert the linearization: recover the grid coordinate mapping to `device_id`.
   *
   * @param device_id A device id.
   *
   * @return The grid coordinate that this mesh maps to `device_id`, or an empty `DomainPoint`
   * (`dim == 0`) if no grid coordinate maps to it.
   */
  [[nodiscard]] DomainPoint grid_point(std::uint64_t device_id) const;

  /**
   * @brief Whether `device_id` is produced by some grid coordinate of this mesh.
   */
  [[nodiscard]] bool contains(std::uint64_t device_id) const;

  /**
   * @brief Whether this mesh is a submesh of `super`.
   *
   * A mesh is a submesh of another when every device it maps to is also mapped by the other. This
   * is used by partition subsumption: a partition can subsume another only if the subsumed
   * partition's mesh is a submesh of the subsuming one.
   */
  [[nodiscard]] bool is_submesh_of(const DeviceMesh& super) const;

  [[nodiscard]] bool operator==(const DeviceMesh& other) const;
  [[nodiscard]] bool operator!=(const DeviceMesh& other) const;

  [[nodiscard]] std::size_t hash() const;
  [[nodiscard]] std::string to_string() const;

 private:
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> shape_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides_{};
  DomainPoint shape_dp_{};
  DomainPoint strides_dp_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/device_mesh.inl>
