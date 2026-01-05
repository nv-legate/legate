/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <string>

namespace legate {

enum class ImageComputationHint : std::uint8_t;

}  // namespace legate

namespace legate::detail {

class Storage;
class TransformStack;

/**
 * @brief A partition constructed by taking the image of a function store.
 *
 * An Image partition maps colors from a partition of a *function store* onto indices of
 * another store. The function store contains either index values or rectangles; rectangles are
 * logically expanded to the set of points they cover. Each color in the function store
 * partition defines the corresponding sub-store in the resulting image partition.
 *
 * Image partitions are generally neither disjoint nor complete. They may be computed either
 * precisely or approximately depending on the provided @ref ImageComputationHint.
 */
class Image final : public Partition {
 public:
  /**
   * @brief Construct an `Image` partition.
   *
   * An Image partition of store A is derived from a partition of store B by interpreting indices
   * contained in B as a function from B's domain to A's domain (hence the name "image"); store B is
   * called the @a function store of the image partition. The function store may contain rectangles
   * instead of indices, in which case the rectangles are logically expanded to individual points
   * they contain.
   *
   * The following show two examples of Image partitions, one where the function store has indices
   * and one with rectangles.
   *
   * With indices:
   * @code
   *                     color
   *               (0)         (1)
   * function: [0,1,3,4,5], [2,6,7,8]
   *
   * indices:                  0  1  2  3  4  5  6  7  8
   * sub-store for color (0)   *  *     *  *  *
   * sub-store for color (1)         *           *  *  *
   * @endcode
   *
   * With rectangles:
   * @code
   *                          color
   *                  (0)               (1)
   * function: [ [0,1], [3,5] ], [ [2,2], [6,8] ]
   *
   * indices:                  0  1  2  3  4  5  6  7  8
   * sub-store for color (0)   *  *     *  *  *
   * sub-store for color (1)         *           *  *  *
   * @endcode
   *
   * Image partitions are neither disjoint nor complete (i.e., cover the entire store).
   *
   * The image computation is done precisely when the `hint` is
   * legate::ImageComputationHint::NO_HINT, or approximate with the other two values (see
   * legate::ImageComputationHint).
   *
   * @param func The function store
   * @param func_partition the function store's partition to use in image computation
   * @param machine the machine on which the image computation tasks are launched
   * @param hint a hint to the image computation (precise vs. approximate)
   */
  Image(InternalSharedPtr<detail::LogicalStore> func,
        InternalSharedPtr<Partition> func_partition,
        mapping::detail::Machine machine,
        ImageComputationHint hint);

  bool operator==(const Image& other) const;

  /**
   * @copydoc Partition::is_complete_for()
   */
  [[nodiscard]] bool is_complete_for(const detail::Storage& storage) const override;

  /**
   * @copydoc Partition::is_disjoint_for()
   *
   * Image partitions are generally *not* disjoint.
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
   * @throws std::invalid_argument Always.
   */
  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;

  /**
   * @copydoc Partition::bloat()
   *
   * @throws std::invalid_argument Always.
   */
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  /**
   * @brief Construct a Legion logical partition for the target region.
   *
   * @param region The logical region covering this partition.
   * @param complete Whether the partition should be complete.
   *
   * If the function partition has no launch domain, returns
   * ``Legion::LogicalPartition::NO_PART``. Otherwise, constructs a logical partition according
   * to the function store, partition, and hint.
   */
  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  /**
   * @copydoc Partition::has_launch_domain().
   *
   * Delegates to the function partition.
   */
  [[nodiscard]] bool has_launch_domain() const override;

  /**
   * @copydoc Partition::launch_domain().
   *
   * Delegates to the function partition.
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
   * Delegates to the function partition.
   */
  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

  /**
   * @copydoc Partition::convert().
   *
   * @throw NonInvertibleTransformation If the transform isn't the identity.
   */
  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  /**
   * @copydoc Partition::invert()
   *
   * @throw NonInvertibleTransformation If the transform isn't the identity.
   */
  [[nodiscard]] InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  /**
   * @brief Return the function store of this image partition.
   *
   * See legate::detail::Image::Image to learn about the role of function store.
   *
   * @return The function store of this `Image` partition
   */
  [[nodiscard]] const InternalSharedPtr<detail::LogicalStore>& func() const;

 private:
  InternalSharedPtr<detail::LogicalStore> func_;
  InternalSharedPtr<Partition> func_partition_{};
  mapping::detail::Machine machine_{};
  ImageComputationHint hint_{};
};

[[nodiscard]] InternalSharedPtr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                                    InternalSharedPtr<Partition> func_partition,
                                                    mapping::detail::Machine machine,
                                                    ImageComputationHint hint);

}  // namespace legate::detail

#include <legate/partitioning/detail/partition/image.inl>
