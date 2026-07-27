/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>
#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class DataMovementRequests;
class Type;
class Shape;
class Partition;
class TransformStack;

class Storage {
 public:
  /**
   * @brief Construct a Storage object lazily.
   *
   * The backing of the storage could be backed by a region field
   * or a future/future map depending on the shape and whether
   * optimizing for a scalar. If executing with inline task launches,
   * the storage is an inline storage.
   *
   * @param shape The shape of the storage.
   * @param type The type of the storage.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape,
          InternalSharedPtr<Type> type,
          std::string_view provenance);
  /**
   * @brief Construct a Storage object initialized eagerly from an existing scalar.
   *
   * The backing of the storage is a future unless executing with inline task launches,
   * in which case it is an inline storage. The scalar data is copied into the created storage.
   *
   * @param shape The shape of the storage.
   * @param scalar The scalar to copy into the storage.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape, const Scalar& scalar, std::string_view provenance);
  /**
   * @brief Construct a Storage object from an existing external allocation.
   *
   * This constructor is only valid when executing with inline task launches.
   *
   * Changes to the data underlying this storage will be reflected in the external allocation.
   *
   * @param shape The shape of the storage.
   * @param type The type of the storage.
   * @param buffer The buffer viewing the external allocation that backs the storage.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape,
          InternalSharedPtr<Type> type,
          InternalSharedPtr<Buffer> buffer,
          std::string_view provenance);

  Storage(Storage&&) noexcept            = default;
  Storage& operator=(Storage&&) noexcept = default;

  [[nodiscard]] std::uint64_t id() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const;
  [[nodiscard]] const Rect& domain() const;
  [[nodiscard]] const Point& extents() const;
  [[nodiscard]] std::size_t volume() const;
  [[nodiscard]] std::uint32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<Storage>& other) const;
  /**
   * @brief Determine whether two stores refer to the same memory.
   *
   * This routine can be used to determine whether two seemingly unrelated stores refer to the
   * same logical memory region.
   *
   * @param other The `Storage` to compare with.
   *
   * @return `true` if two stores cover the same underlying memory region, `false` otherwise.
   */
  [[nodiscard]] bool equal(const Storage& other) const;
  [[nodiscard]] std::string_view provenance() const;

  [[nodiscard]] std::tuple<InternalSharedPtr<Buffer>, Rect, bool> map(
    const Rect& domain,
    const InternalSharedPtr<Partition>& partition,
    const Point& color,
    legate::mapping::StoreTarget target,
    bool is_read,
    bool is_transformed,
    DataMovementRequests* requests);
  [[nodiscard]] std::pair<InternalSharedPtr<Buffer>, Rect> map_reduction(
    const Rect& domain,
    const InternalSharedPtr<Partition>& partition,
    const Point& color,
    legate::mapping::StoreTarget target,
    bool is_parallel,
    const InternalSharedPtr<Type>& type,
    ReductionOpKind op_kind,
    bool store_has_identity,
    DataMovementRequests* requests);

  void unmap();
  void free_early();

  [[nodiscard]] mapping::StoreTarget target() const;

  /**
   * @brief Replaces the storage's canonical backing with the given buffer.
   *
   * @param storage_data The buffer that becomes the canonical backing.
   */
  void set_backing_buffer(const InternalSharedPtr<Buffer>& storage_data);

  /**
   * @brief Binds a previously unbound storage to a concrete buffer and (zero-based) domain.
   *
   * Valid only when this storage is unbound. In single-rank execution the whole store is produced
   * at once, so this immediately finishes the binding: the shape becomes READY with extents derived
   * from @p domain, its domain is set, its backing `Buffer` is @p storage_data, and it is
   * given a trivial (no-)partition. In multi-rank execution the rank only produced its own piece,
   * so this records the local piece but leaves the storage "pending" (still unbound); the global
   * shape and the (Opaque) key partition are set later by `finalize_unbound_partitioned` once every
   * rank has bound a buffer.
   *
   * @param storage_data The buffer produced by the task.
   * @param domain The zero-based domain (extents) of the produced piece.
   */
  void bind(const InternalSharedPtr<Buffer>& storage_data, const Rect& domain);

  void finalize_unbound();

  /**
   * @brief Complete a multi-rank unbound binding once all ranks' pieces are known.
   *
   * Sets the global shape, re-bases this rank's local piece into its global sub-domain, records the
   * whole-store domain, and installs the given (Opaque) key partition.
   *
   * @param global_extents      The stitched-together global extents of the store.
   * @param shard_domain This rank's shard expressed in global coordinates.
   * @param key_partition       The completed Opaque partition describing all ranks' pieces.
   */
  void finalize_unbound_partitioned(Point global_extents,
                                    Rect shard_domain,
                                    InternalSharedPtr<Partition> key_partition);

  [[nodiscard]] Restrictions compute_restrictions() const;
  [[nodiscard]] std::optional<InternalSharedPtr<Partition>> find_key_partition(
    const Restrictions& restrictions) const;
  void set_key_partition(InternalSharedPtr<Partition> key_partition);
  void reset_key_partition() noexcept;

  [[nodiscard]] std::string to_string() const;

 private:
  std::uint64_t storage_id_{};
  InternalSharedPtr<Shape> shape_{};
  Rect domain_{};
  InternalSharedPtr<Type> type_{};
  std::string_view provenance_{};  // only relevant on the root Storage

  // The buffer directly backing this storage. Its domain coincides with `domain_` for a single-rank
  // store, but for a multi-rank unbound store it is this rank's local (global-coordinate) piece,
  // which is a sub-domain of `domain_`. Null until the storage is materialized (an unbound store
  // has no backing until a buffer is bound; a lazily-created bound store has none until first
  // written).
  InternalSharedPtr<Buffer> buffer_{};
  InternalSharedPtr<Partition> key_partition_{};
};

}  // namespace legate::detail

#include <legate/data/detail/storage.inl>
