/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
   * @copydoc Partition::is_complete_for()
   *
   * Always returns true, since the entire store belongs to a single partition.
   */
  [[nodiscard]] bool is_complete_for(const detail::Storage& /*storage*/) const override;

  /**
   * @copydoc Partition::is_disjoint_for()
   *
   * Always returns true, as there is only one subregion.
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
   * @copydoc Partition::construct()
   *
   * Constructs a degenerate Legion partition that simply wraps the entire region.
   */
  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion /*region*/,
                                                   bool /*complete*/) const override;

  /**
   * @copydoc Partition::has_launch_domain()
   *
   * Always false; a `NoPartition` has no launch domain.
   */
  [[nodiscard]] bool has_launch_domain() const override;

  /**
   * @copydoc Partition::launch_domain()
   *
   * Undefined for `NoPartition`, always throws the exception.
   *
   * @throws std::invalid_argument
   */
  [[nodiscard]] Domain launch_domain() const override;

  /**
   * @copydoc Partition::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

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

[[nodiscard]] InternalSharedPtr<NoPartition> create_no_partition();

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/no_partition.inl>
