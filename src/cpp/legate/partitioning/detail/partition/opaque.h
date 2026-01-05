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

#include <cstddef>
#include <cstdint>
#include <string>

namespace legate::detail {

class Storage;

/**
 * @brief Opaque partitions are primarily used for representing partition information
 *        of unbounded stores.
 */
class Opaque final : public Partition {
 public:
  /**
   * @brief Construct an `Opaque` partition.
   *
   * An opaque partition represents the partitions of unbounded stores. We can construct this
   * partition without having concrete (completely constructed) IndexSpace and IndexPartition, hence
   * the name "opaque". This is useful to make progress in the cases where the same partition will
   * be applied to some other store or may be used as a key partition. The nature of unbounded store
   * is such that the bounds are only known when the task producing the store actually completes
   * execution. By having a opaque partition we can still make progress without having to wait for
   * all the details of the partition being available, which happens only after the task executes.
   *
   * @param ispace The IndexSpace for this partition (name only).
   * @param ipartition The IndexPartition (name only).
   * @param color_domain The domain of the partition.
   */
  Opaque(Legion::IndexSpace ispace, Legion::IndexPartition ipartition, const Domain& color_domain);

  bool operator==(const Opaque& other) const;
  bool operator<(const Opaque& other) const;

  /**
   * @brief Indicate if the partition is disjoint for a given launch domain
   */
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  /**
   * @brief Indicate if the partition is convertible. Always return false.
   *
   * The opacity of the partition makes it non-convertible. For converting we
   * need access to the shape of the partition and that is not available for
   * Opaque partitions.
   */
  [[nodiscard]] bool is_convertible() const override;
  /**
   * @brief Indicate if the partition is invertible. Always return false.
   *
   * The opacity of the partition makes it non-invertible. For inverting we
   * need access to the shape of the partition and that is not available for
   * Opaque partitions.
   */
  [[nodiscard]] bool is_invertible() const override;
  /**
   * @brief Indicate if the partition covers a given storage. Always return true, as Opaque
   * partitions are created only for unbound stores and thus are complete by construction.
   */
  [[nodiscard]] bool is_complete_for(const detail::Storage& /*storage*/) const override;
  /**
   * @brief Scale the partition by given factors. Not implemented.
   */
  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;
  /**
   * @brief Bloat each chunk in the partition by given offsets. Not implemented.
   */
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;
  /**
   * @brief Construct a Legion logical partition for a given Legion logical region.
   *
   * @param region The region we're trying to partition.
   * @param complete To indicate if the partition is complete or not.
   */
  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;
  /**
   * @brief Indicate if the partition's color shape can be converted into a launch domain.
   * Always return true.
   */
  [[nodiscard]] bool has_launch_domain() const override;
  /**
   * @brief Convert the partition's color shape into a launch domain.
   */
  [[nodiscard]] Domain launch_domain() const override;
  /**
   * @brief Return a human-readable representation of the partition in a string
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @copydoc Partition::has_color_shape().
   */
  [[nodiscard]] bool has_color_shape() const override;

  /**
   * @brief Return the partition's color shape
   */
  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;
  /**
   * @brief Convert the partition using a given transformation stack. Raise
   * runtime_error unless the transformation is the identity.
   *
   * Only identity works as we can not access information like the shape of
   * an Opaque transformation.
   *
   * @param self       A shared pointer to this partition.
   * @param transform  The transformation stack to apply.
   */
  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;
  /**
   * @brief Invert the partition using a given transformation stack. Raise
   * NonInvertibleTransformation unless the transformation is the identity.
   *
   * Only identity works as we can not access information like the shape of
   * an Opaque transformation.
   *
   * @param self       A shared pointer to this partition.
   * @param transform  The transformation stack to apply.
   */
  [[nodiscard]] InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  Opaque(const Opaque&)                = default;
  Opaque& operator=(const Opaque&)     = default;
  Opaque(Opaque&&) noexcept            = default;
  Opaque& operator=(Opaque&&) noexcept = default;

 private:
  Legion::IndexSpace ispace_{};
  Legion::IndexPartition ipartition_{};
  Domain color_domain_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape_{};
};

/*
 * @brief Create an Opaque Partition.
 *
 * @param ispace The name of the Index Space.
 * @param iparam The name of the Partition.
 * @param color_domain The domain of the partition.
 *
 * @return The Opaque partition pointer.
 */
[[nodiscard]] InternalSharedPtr<Opaque> create_opaque(Legion::IndexSpace ispace,
                                                      Legion::IndexPartition ipartition,
                                                      const Domain& color_domain);

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/opaque.inl>
