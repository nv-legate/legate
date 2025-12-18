/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>
#include <string>

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
  enum class Kind : std::uint8_t {
    NO_PARTITION,
    TILING,
    OPAQUE,
    IMAGE,
  };

  Partition()                                = default;
  virtual ~Partition()                       = default;
  Partition(const Partition&)                = default;
  Partition(Partition&&) noexcept            = default;
  Partition& operator=(const Partition&)     = default;
  Partition& operator=(Partition&&) noexcept = default;

  /**
   * @brief Returns the specific kind of this partition.
   * @return The partition kind.
   */
  [[nodiscard]] virtual Kind kind() const = 0;

  /**
   * @brief Checks whether this partition fully covers the given storage.
   *
   * @param storage The storage to check completeness against.
   * @return True if the partition completely covers the storage.
   */
  [[nodiscard]] virtual bool is_complete_for(const detail::Storage& storage) const = 0;

  /**
   * @brief Checks whether this partition is disjoint for the given launch domain.
   *
   * @param launch_domain The domain of tasks or points launched.
   * @return True if all subregions are disjoint within the provided domain.
   */
  [[nodiscard]] virtual bool is_disjoint_for(const Domain& launch_domain) const = 0;

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
   * @brief Constructs a Legion logical partition from this partition description.
   *
   * @param region   The parent logical region.
   * @param complete Whether Legion should consider this partition complete.
   * @return The created LogicalPartition.
   */
  // NOLINTNEXTLINE(google-default-arguments)
  [[nodiscard]] virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                           bool complete = false) const = 0;

  /**
   * @brief Returns whether this partition defines a launch domain.
   *
   * @return True if a launch domain exists.
   */
  [[nodiscard]] virtual bool has_launch_domain() const = 0;

  /**
   * @brief Returns the launch domain associated with this partition.
   *
   * @return The launch domain.
   *
   * @note Only valid if has_launch_domain() is true.
   */
  [[nodiscard]] virtual Domain launch_domain() const = 0;

  /**
   * @brief Returns a human-readable string describing the partition.
   *
   * @return A string representation.
   */
  [[nodiscard]] virtual std::string to_string() const = 0;

  /**
   * @brief Returns the shape (size) of the partition's color space.
   *
   * @return A span of dimension sizes.
   */
  [[nodiscard]] virtual Span<const std::uint64_t> color_shape() const = 0;

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
