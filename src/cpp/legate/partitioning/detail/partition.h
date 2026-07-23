/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace legate::detail {

class LogicalStore;
class Storage;
class TransformStack;
class Restrictions;

/**
 * @brief Abstract base class representing a partitioning strategy.
 *
 * A Partition describes how an index space or region is divided into subregions.
 * Implementations define the partition kind, how to check validity and constraints,
 * and how to construct Legion partitions or transform them.
 */
class Partition {
 public:
  Partition()                                = default;
  virtual ~Partition()                       = default;
  Partition(const Partition&)                = default;
  Partition(Partition&&) noexcept            = default;
  Partition& operator=(const Partition&)     = default;
  Partition& operator=(Partition&&) noexcept = default;

  /**
   * @brief Indicates whether this partition can be converted (via transform).
   *
   * @return True if the partition supports conversion.
   */
  [[nodiscard]] virtual bool is_convertible() const = 0;

  /**
   *
   * @brief Indicate whether this partition is invertible or not.
   *
   * A partition is invertible when the partition belongs to a transformed store
   * and the partition can be changed to that of the source store (that was transformed).
   */
  [[nodiscard]] virtual bool is_invertible() const = 0;

  /**
   * @brief Produces a scaled version of this partition.
   *
   * @param factors Scale factors applied to each dimension.
   * @return A new Partition instance representing the scaled partition.
   */
  [[nodiscard]] virtual InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const = 0;

  /**
   * @brief Expands (bloats) the partition by applying offsets to its bounds.
   *
   * @param low_offsets  Offsets applied to the lower bounds.
   * @param high_offsets Offsets applied to the upper bounds.
   * @return A new bloated Partition instance.
   */
  [[nodiscard]] virtual InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const = 0;

  /**
   * @brief Returns whether this partition defines a color space. In effect, this determines
   * whether a call to `color_shape()` is well-formed. The overwhelming majority of partitions
   * are expected to have a color shape.
   *
   * @return `true` if the partition has a color shape, `false` otherwise..
   */
  [[nodiscard]] virtual bool has_color_shape() const = 0;

  /**
   * @brief Returns the shape (size) of the partition's color space.
   *
   * @return A point whose coordinates are the color-space extents.
   */
  [[nodiscard]] virtual const Point& color_shape() const = 0;

  [[nodiscard]] virtual bool is_equi_size(const Rect& domain) const = 0;

  /**
   * @brief Returns the launch domain associated with this partition.
   *
   * @return The launch domain.
   *
   * @note Only valid if has_launch_domain() is true.
   */
  [[nodiscard]] virtual Rect launch_domain() const = 0;

  [[nodiscard]] virtual Point compute_color(std::uint64_t device_id) const = 0;

  /**
   * @brief Compute the device ids a given color is mapped to.
   *
   * A color may map to more than one device id when the partition replicates the piece.
   *
   * @param color The piece color.
   *
   * @return The device ids holding the piece with the given color.
   */
  [[nodiscard]] virtual std::vector<std::uint64_t> compute_device_ids(const Point& color) const = 0;

  [[nodiscard]] virtual Rect shard(const Rect& domain, const Point& color) const = 0;

  [[nodiscard]] virtual bool subsumes(const Rect& domain,
                                      const InternalSharedPtr<Partition>& other) const = 0;

  /**
   * @brief Returns a human-readable string describing the partition.
   *
   * @return A string representation.
   */
  [[nodiscard]] virtual std::string to_string() const = 0;

  /**
   * @brief Converts this partition using the given transform stack.
   *
   * @param self       A shared pointer to this partition.
   * @param transform  The transformation stack to apply.
   * @return A new converted Partition.
   */
  [[nodiscard]] virtual InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const = 0;

  /**
   * @brief Applies the inverse of the given transform to this partition.
   *
   * @param self       A shared pointer to this partition.
   * @param transform  The transformation stack whose inverse is applied.
   * @return A new inverted Partition.
   */
  [[nodiscard]] virtual InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const = 0;
};

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate::detail
