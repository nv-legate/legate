/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <string>
#include <vector>

namespace legate::detail {

class Storage;
class TransformStack;

/**
 * @brief A trivial partition representing the absence of any subdivision.
 *
 * `NoPartition` indicates that the store is treated as a single, unpartitioned
 * region. It is always complete, always disjoint (trivially), and does not
 * define a launch domain. Transformations such as scaling or bloating return
 * an equivalent `NoPartition`.
 */
class NoPartition final : public Partition {
 public:
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
   *
   * Returns another `NoPartition`, since scaling has no effect.
   */
  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;

  /**
   * @copydoc Partition::bloat()
   *
   * Returns another `NoPartition`, since bloating does not change the structure.
   */
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  /**
   * @copydoc Partition::has_color_shape().
   */
  [[nodiscard]] bool has_color_shape() const override;

  /**
   * @copydoc Partition::color_shape()
   *
   * Always throws the exception.
   *
   * @throws std::invalid_argument
   */
  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

  [[nodiscard]] bool is_equi_size(const Domain& domain) const override;

  /**
   * @copydoc Partition::launch_domain()
   *
   * Undefined for `NoPartition`, always throws the exception.
   *
   * @throws std::invalid_argument
   */
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] DomainPoint compute_color(std::uint64_t device_id) const override;

  /**
   * @copydoc Partition::compute_device_ids()
   *
   * A `NoPartition` places the whole (single) region on every device, so this returns every device
   * id in the machine.
   */
  [[nodiscard]] std::vector<std::uint64_t> compute_device_ids(
    const DomainPoint& color) const override;

  [[nodiscard]] Domain shard(const Domain& domain, const DomainPoint& color) const override;

  [[nodiscard]] bool subsumes(const Domain& domain,
                              const InternalSharedPtr<Partition>& other) const override;

  /**
   * @copydoc Partition::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

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
};

[[nodiscard]] const InternalSharedPtr<Partition>& create_no_partition();

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/no_partition.inl>
