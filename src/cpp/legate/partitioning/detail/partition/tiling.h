/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/device_mesh.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace legate::detail {

class Storage;

/**
 * @brief A partition that divides an index space into regularly shaped tiles.
 *
 * A `Tiling` represents an N-dimensional grid of tiles, each tile having a fixed
 * `tile_shape` and arranged according to a specified `color_shape`. Optional
 * `offsets` and `strides` shift and space tiles within the parent store`s
 * coordinate space.
 *
 * Tiled partitions are typically disjoint and complete when the parent extents
 * are compatible with the tile/offset/stride configuration, but may produce
 * clipped or empty tiles near boundaries.
 */
class Tiling final : public Partition {
 public:
  /**
   * @brief Construct a `Tiling` partition.
   *
   * An example of a Tiling we can create, over a 1d index space:
   *
   * @code
   * Tiling(tile_shape=(3,), color_shape=(4,), offsets=(1,))
   *
   *                          offset
   *                          V
   * indices:              0  1  2  3  4  5  6  7  8  9 10 11 12
   * tile for color (0,)      *  *  *
   * tile for color (1,)               *  *  *
   * tile for color (2,)                        *  *  *
   * tile for color (3,)                                 *  *  *
   * @endcode
   *
   * This formulation is somewhat overconstrained. In theory, you could deduce `color_shape`
   * from `tile_shape` when applying the tiling to a given store.
   *
   * @param tile_shape The size that each sub-tile must be.
   * @param color_shape The number of colors in each dimension.
   * @param offsets The number of entries to skip per dimension.
   */
  Tiling(DomainPoint tile_shape, DomainPoint color_shape, DomainPoint offsets);

  /**
   * @brief Construct a Tiling partition with explicit strides.
   *
   * Strides specify the spacing between successive tiles in each dimension.
   * If omitted, the stride defaults to the tile size.
   *
   * @param tile_shape  The size of each tile.
   * @param color_shape The number of tiles in each dimension.
   * @param offsets     Origin offsets for the first tile.
   * @param strides     Spacing between tiles.
   */
  Tiling(DomainPoint tile_shape, DomainPoint color_shape, DomainPoint offsets, DomainPoint strides);

  /**
   * @brief Construct a Tiling partition with an explicit device mesh.
   *
   * The `color_to_mesh` mapping associates dimensions of the color shape with dimensions of the
   * device mesh (`color_to_mesh[c]` is the mesh dimension driven by color dimension `c`, or `-1`
   * if color dimension `c` does not select a mesh dimension). Mesh dimensions not selected by any
   * color dimension are replication axes: a single tile is placed on every device along them.
   *
   * @param tile_shape    The size of each tile.
   * @param color_shape   The number of tiles in each dimension.
   * @param offsets       Origin offsets for the first tile.
   * @param strides       Spacing between tiles.
   * @param device_mesh   The device mesh the tiles are mapped onto.
   * @param color_to_mesh Mapping from color dimensions to mesh dimensions (`-1` for unmapped).
   */
  Tiling(DomainPoint tile_shape,
         DomainPoint color_shape,
         DomainPoint offsets,
         DomainPoint strides,
         DeviceMesh device_mesh,
         SmallVector<std::int32_t, LEGATE_MAX_DIM> color_to_mesh);

  bool operator==(const Tiling& other) const;

  /**
   * @copydoc Partition::is_convertible()
   */
  [[nodiscard]] bool is_convertible() const override;

  /**
   * @brief Indicate whether this partition is invertible or not.
   */
  [[nodiscard]] bool is_invertible() const override;

  /**
   * @copydoc Partition::scale()
   */
  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;

  /**
   * @copydoc Partition::blaot()
   */
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  /**
   * @copydoc Partition::has_color_shape().
   */
  [[nodiscard]] bool has_color_shape() const override;

  /**
   * @copydoc Partition::color_shape()
   */
  [[nodiscard]] const DomainPoint& color_shape() const override;

  [[nodiscard]] bool is_equi_size(const Domain& domain) const override;

  /**
   * @copydoc Partition::launch_domain()
   */
  [[nodiscard]] Domain launch_domain() const override;

  /**
   * @copydoc Partition::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the tile shape used in this tiling.
   *
   * @return A point describing the tile size per dimension.
   */
  [[nodiscard]] const DomainPoint& tile_shape() const;

  /**
   * @copydoc Partition::convert()
   */
  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  /**
   * @copydoc Partition::invert()
   */
  [[nodiscard]] InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  [[nodiscard]] DomainPoint compute_color(std::uint64_t device_id) const override;

  /**
   * @brief Compute the device ids a given tile color is mapped to.
   *
   * A color maps to more than one device id when the tiling replicates the tile (i.e. the device
   * mesh has a dimension not selected by any color dimension).
   *
   * @param color The tile color.
   *
   * @return The device ids holding the tile.
   */
  [[nodiscard]] std::vector<std::uint64_t> compute_device_ids(
    const DomainPoint& color) const override;

  /**
   * @brief Return the device mesh the tiles are mapped onto.
   */
  [[nodiscard]] const DeviceMesh& device_mesh() const;

  [[nodiscard]] Domain shard(const Domain& domain, const DomainPoint& color) const override;

  [[nodiscard]] bool subsumes(const Domain& domain,
                              const InternalSharedPtr<Partition>& other) const override;

  /**
   * @brief Return the offset of the first tile in each dimension.
   *
   * @return A point of per-dimension offsets.
   */
  [[nodiscard]] const DomainPoint& offsets() const;

  /**
   * @brief Return the stride (spacing) between tiles.
   *
   * @return A point of per-dimension strides.
   */
  [[nodiscard]] const DomainPoint& strides() const;

  /**
   * @brief Compute a hash value for this tiling configuration.
   *
   * Useful for caching or map-based structures that key on tilings.
   *
   * @return A hash of tile shape, color shape, offsets, strides, and overlap status.
   */
  [[nodiscard]] std::size_t hash() const;

 private:
  void init_device_mesh_from_color_shape_();

  DomainPoint tile_shape_{};
  DomainPoint color_shape_{};
  DomainPoint offsets_{};
  DomainPoint strides_{};
  DeviceMesh device_mesh_{};
  // color_to_mesh_[c] is the mesh dimension selected by color dimension c, or -1 if unmapped.
  SmallVector<std::int32_t, LEGATE_MAX_DIM> color_to_mesh_{};
};

[[nodiscard]] InternalSharedPtr<Partition> create_tiling(DomainPoint tile_shape,
                                                         DomainPoint color_shape,
                                                         DomainPoint offsets = {});

[[nodiscard]] InternalSharedPtr<Partition> create_tiling(DomainPoint tile_shape,
                                                         DomainPoint color_shape,
                                                         DomainPoint offsets,
                                                         DomainPoint strides);

[[nodiscard]] InternalSharedPtr<Partition> create_tiling(
  DomainPoint tile_shape,
  DomainPoint color_shape,
  DomainPoint offsets,
  DomainPoint strides,
  DeviceMesh device_mesh,
  SmallVector<std::int32_t, LEGATE_MAX_DIM> color_to_mesh);

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/tiling.inl>
