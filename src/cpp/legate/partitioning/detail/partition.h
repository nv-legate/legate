/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <iosfwd>
#include <memory>
#include <string>

namespace legate::detail {

class LogicalStore;
class Storage;

class Partition {
 public:
  enum class Kind : std::uint8_t {
    NO_PARTITION,
    TILING,
    WEIGHTED,
    IMAGE,
  };

  Partition()                                = default;
  virtual ~Partition()                       = default;
  Partition(const Partition&)                = default;
  Partition(Partition&&) noexcept            = default;
  Partition& operator=(const Partition&)     = default;
  Partition& operator=(Partition&&) noexcept = default;

  [[nodiscard]] virtual Kind kind() const = 0;

  [[nodiscard]] virtual bool is_complete_for(const detail::Storage& storage) const          = 0;
  [[nodiscard]] virtual bool is_disjoint_for(const Domain& launch_domain) const             = 0;
  [[nodiscard]] virtual bool satisfies_restrictions(const Restrictions& restrictions) const = 0;
  [[nodiscard]] virtual bool is_convertible() const                                         = 0;

  [[nodiscard]] virtual InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const = 0;
  [[nodiscard]] virtual InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const = 0;

  // NOLINTNEXTLINE(google-default-arguments)
  [[nodiscard]] virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                           bool complete = false) const = 0;

  [[nodiscard]] virtual bool has_launch_domain() const = 0;
  [[nodiscard]] virtual Domain launch_domain() const   = 0;

  [[nodiscard]] virtual std::string to_string() const = 0;

  [[nodiscard]] virtual Span<const std::uint64_t> color_shape() const = 0;

  [[nodiscard]] virtual InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const = 0;
  [[nodiscard]] virtual InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const = 0;
};

class NoPartition : public Partition {
 public:
  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage& /*storage*/) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& /*restrictions*/) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion /*region*/,
                                                   bool /*complete*/) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;
  [[nodiscard]] InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;
};

class Tiling : public Partition {
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
  Tiling(SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
         SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
         SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
         SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides);

  bool operator==(const Tiling& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage& storage) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] Span<const std::uint64_t> tile_shape() const;
  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;
  [[nodiscard]] InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  [[nodiscard]] Span<const std::int64_t> offsets() const;
  [[nodiscard]] Span<const std::uint64_t> strides() const;
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
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> get_child_offsets(
    Span<const std::uint64_t> color) const;

  [[nodiscard]] std::size_t hash() const;

 private:
  bool overlapped_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape_{};
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides_{};
};

class Weighted : public Partition {
 public:
  Weighted(Legion::FutureMap weights, const Domain& color_domain);
  ~Weighted() override;

  bool operator==(const Weighted& other) const;
  bool operator<(const Weighted& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;
  [[nodiscard]] bool is_complete_for(const detail::Storage& /*storage*/) const override;

  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;
  [[nodiscard]] InternalSharedPtr<Partition> invert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;

  Weighted(const Weighted&)                = default;
  Weighted& operator=(const Weighted&)     = default;
  Weighted(Weighted&&) noexcept            = default;
  Weighted& operator=(Weighted&&) noexcept = default;

 private:
  Legion::FutureMap weights_{};
  Domain color_domain_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape_{};
};

class Image : public Partition {
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

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage& storage) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] InternalSharedPtr<Partition> scale(
    Span<const std::uint64_t> factors) const override;
  [[nodiscard]] InternalSharedPtr<Partition> bloat(
    Span<const std::uint64_t> low_offsets, Span<const std::uint64_t> high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] Span<const std::uint64_t> color_shape() const override;

  [[nodiscard]] InternalSharedPtr<Partition> convert(
    const InternalSharedPtr<Partition>& self,
    const InternalSharedPtr<TransformStack>& transform) const override;
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

[[nodiscard]] InternalSharedPtr<NoPartition> create_no_partition();

[[nodiscard]] InternalSharedPtr<Tiling> create_tiling(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets = {});

[[nodiscard]] InternalSharedPtr<Tiling> create_tiling(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides);

[[nodiscard]] InternalSharedPtr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                                          const Domain& color_domain);

[[nodiscard]] InternalSharedPtr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                                    InternalSharedPtr<Partition> func_partition,
                                                    mapping::detail::Machine machine,
                                                    ImageComputationHint hint);

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate::detail

#include <legate/partitioning/detail/partition.inl>
