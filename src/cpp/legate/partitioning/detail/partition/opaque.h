/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace legate::detail {

class Storage;
class TransformStack;

/**
 * @brief A partition made of an irregular set of rank-local pieces (an "opaque" tiling).
 *
 * `Opaque` partitions represent the partitioning of unbound stores. Unlike `Tiling`, the pieces
 * need not be regularly shaped: each color of the partition maps to an arbitrary rank-local
 * `Domain` placed at an explicit global offset. Concretely, an `Opaque` is a mapping from colors
 * to global sub-`Domain`s (one per producing rank), together with the color space shape.
 *
 * The nature of an unbound store is such that the shape (and therefore the concrete per-color
 * `Domain`s) is only known once the task producing the store finishes executing. An `Opaque` is
 * therefore created as a *placeholder* carrying just the color-space shape (so the launch domain
 * can be resolved during partitioning), and later completed via `fill()` from the all-gathered
 * per-rank domains (see the unbound-store post-processing in the task launch path).
 */
class Opaque final : public Partition {
 public:
  /**
   * @brief Construct a placeholder `Opaque` partition.
   *
   * Only the color-space shape is known at construction time (it is derived from the launch domain
   * of the producing task). The concrete per-color domains are filled in later via `fill()`.
   *
   * @param color_shape The number of colors (pieces) in each dimension of the color space.
   */
  explicit Opaque(DomainPoint color_shape);

  bool operator==(const Opaque& other) const;

  /**
   * @copydoc Partition::is_convertible()
   */
  [[nodiscard]] bool is_convertible() const override;

  /**
   * @copydoc Partition::is_invertible()
   */
  [[nodiscard]] bool is_invertible() const override;

  /**
   * @copydoc Partition::scale()
   *
   * Not supported for `Opaque` partitions.
   */
  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;

  /**
   * @copydoc Partition::bloat()
   *
   * Not supported for `Opaque` partitions.
   */
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  /**
   * @copydoc Partition::has_color_shape()
   */
  [[nodiscard]] bool has_color_shape() const override;

  /**
   * @copydoc Partition::color_shape()
   */
  [[nodiscard]] const DomainPoint& color_shape() const override;

  /**
   * @copydoc Partition::is_equi_size()
   *
   * Always `false`: an opaque partition is assumed to have irregular pieces, so copies out of it
   * always take the general shuffle path rather than the equal-size all-gather fast path.
   */
  [[nodiscard]] bool is_equi_size(const Domain& domain) const override;

  /**
   * @copydoc Partition::launch_domain()
   */
  [[nodiscard]] Domain launch_domain() const override;

  /**
   * @copydoc Partition::compute_color()
   */
  [[nodiscard]] DomainPoint compute_color(std::uint64_t device_id) const override;

  /**
   * @copydoc Partition::compute_device_ids()
   */
  [[nodiscard]] std::vector<std::uint64_t> compute_device_ids(
    const DomainPoint& color) const override;

  /**
   * @copydoc Partition::shard()
   *
   * Returns the global sub-domain of the piece with the given color. The domain argument is
   * ignored; opaque partitions store absolute global sub-domains.
   */
  [[nodiscard]] Domain shard(const Domain& domain, const DomainPoint& color) const override;

  /**
   * @copydoc Partition::subsumes()
   *
   * An opaque partition only subsumes itself (an identical opaque distribution).
   */
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

  /**
   * @brief Complete the placeholder with the concrete per-rank pieces.
   *
   * @param colors  `colors[d]` is the color of the piece produced by device (rank) `d`.
   * @param domains `domains[d]` is that piece's global sub-`Domain`.
   */
  void fill(SmallVector<DomainPoint> colors, SmallVector<Domain> domains);

  /**
   * @brief Whether `fill()` has been called (the per-color domains are known).
   */
  [[nodiscard]] bool filled() const;

 private:
  [[nodiscard]] std::optional<std::uint64_t> device_of_(const DomainPoint& color) const;

  DomainPoint color_shape_{};
  // Indexed by device (rank) id.
  SmallVector<DomainPoint> colors_{};
  SmallVector<Domain> domains_{};
  bool filled_{};
};

[[nodiscard]] InternalSharedPtr<Opaque> create_opaque(DomainPoint color_shape);

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/opaque.inl>
