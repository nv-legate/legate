/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <string>

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
  Tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
         SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
         SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);

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
  Tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
         SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
         SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
         SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides);

  bool operator==(const Tiling& other) const;

  /**
   * @copydoc Partition::is_complete_for()
   *
   * A tiling is complete only if its tiles collectively cover the given storage exactly.
   */
  [[nodiscard]] bool is_complete_for(const detail::Storage& storage) const override;

  /**
   * @copydoc Partition::is_disjoint_for()
   *
   * A tiling is disjoint unless tiles overlap due to offsets or strides.
   */
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;

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
   * @copydoc Partition::construct()
   */
  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  /**
   * @copydoc Partition::has_launch_domain()
   *
   * Always has a launch domain.
   */
  [[nodiscard]] bool has_launch_domain() const override;

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
   * @return A span describing the tile size per dimension.
   */
  [[nodiscard]] Span<const std::uint64_t> tile_shape() const;

  /**
   * @copydoc Partition::has_color_shape().
   */
  [[nodiscard]] bool has_color_shape() const override;

  /**
   * @copydoc Partition::color_shape()
   */
  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

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

  /**
   * @brief Return the offset of the first tile in each dimension.
   *
   * @return A span of per-dimension offsets.
   */
  [[nodiscard]] Span<const std::int64_t> offsets() const;

  /**
   * @brief Return the stride (spacing) between tiles.
   *
   * @return A span of per-dimension strides.
   */
  [[nodiscard]] Span<const std::uint64_t> strides() const;

  /**
   * @brief Check whether this tiling contains a given color coordinate.
   *
   * @param color A multi-dimensional index into the color grid.
   * @return True if the color is within the tiling's color shape.
   */
  [[nodiscard]] bool has_color(Span<const std::uint64_t> color) const;

  /**
   * @brief Apply the tiling to the extents of a parent Store and retrieve the extents of a
   * particular child Store.
   *
   * If `extents` is smaller than, or not otherwise compatible with the parent extents for
   * which this `Tiling` was constructed, then the resultant child extents are "clipped" (size
   * 0).
   *
   * For example, given the example detailed in the constructor (a size 13 parent), suppose we
   * map this to a 6-element substore. There is no way to split 6 elements into 4 sets of 3
   * elements. In that case the children would have extents 3, 2, 0 and 0.
   *
   * `color` specifies the `i, (j, k, ...)`-th color within the tiling, corresponding to a
   * particular child. So, for example for a tiling which produces:
   *
   * @code
   * A B
   * C D
   * @endcode
   *
   * passing `color = (1, 1)` gives you `D`.
   *
   * @param extents The extents of the region on which to apply the tiling.
   * @param color The color corresponding to the child Store to retrieve.
   *
   * @return The sub-region of `extents` corresponding to `color`.
   */
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> get_child_extents(
    Span<const std::uint64_t> extents, Span<const std::uint64_t> color) const;

  /**
   * @brief Compute the offset origin of a child tile within the parent.
   *
   * @param color The tile color.
   *
   * @return The per-dimension offset of the tile.
   */
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> get_child_offsets(
    Span<const std::uint64_t> color) const;

  /**
   * @brief Compute a hash value for this tiling configuration.
   *
   * Useful for caching or map-based structures that key on tilings.
   *
   * @return A hash of tile shape, color shape, offsets, strides, and overlap status.
   */
  [[nodiscard]] std::size_t hash() const;

 private:
  bool overlapped_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape_{};
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides_{};
};

[[nodiscard]] InternalSharedPtr<Tiling> create_tiling(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets = {});

[[nodiscard]] InternalSharedPtr<Tiling> create_tiling(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides);

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/tiling.inl>
