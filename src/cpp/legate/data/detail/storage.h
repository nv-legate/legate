/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
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
class ExternalAllocation;

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
   * @param field_size The size of the field.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape, std::uint32_t field_size, std::string_view provenance);
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
   * @param allocation The external allocation that holds the pointer backing the storage.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape,
          std::uint32_t field_size,
          InternalSharedPtr<ExternalAllocation> allocation,
          std::string_view provenance);

  Storage(Storage&&) noexcept            = default;
  Storage& operator=(Storage&&) noexcept = default;

  [[nodiscard]] std::uint64_t id() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const;
  [[nodiscard]] const Domain& domain() const;
  [[nodiscard]] Span<const std::uint64_t> extents() const;
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
  [[nodiscard]] bool is_mapped() const;

  void set_bound(bool bound);

  [[nodiscard]] std::tuple<InternalSharedPtr<InlineStorage>, Domain, bool> map(
    const Domain& domain,
    const InternalSharedPtr<Partition>& partition,
    const DomainPoint& color,
    legate::mapping::StoreTarget target,
    bool is_read,
    bool is_transformed,
    DataMovementRequests* requests);
  [[nodiscard]] std::pair<InternalSharedPtr<InlineStorage>, Domain> map_reduction(
    const Domain& domain,
    const InternalSharedPtr<Partition>& partition,
    const DomainPoint& color,
    legate::mapping::StoreTarget target,
    bool is_parallel,
    const InternalSharedPtr<Type>& type,
    ReductionOpKind op_kind,
    bool store_has_identity,
    DataMovementRequests* requests);

  void unmap();
  void free_early();

  [[nodiscard]] mapping::StoreTarget target() const;

  void set_inline_storage(InternalSharedPtr<InlineStorage> storage_data);

  /**
   * @brief Binds a previously unbound storage to a concrete inline storage and (zero-based) domain.
   *
   * Valid only when this storage is unbound. In single-rank execution the whole store is produced
   * at once, so this immediately finishes the binding: the shape becomes READY with extents derived
   * from @p domain, its domain is set, its backing `InlineStorage` is @p storage_data, and it is
   * given a trivial (no-)partition. In multi-rank execution the rank only produced its own piece,
   * so this records the local piece but leaves the storage "pending" (still unbound); the global
   * shape and the (Opaque) key partition are set later by `finalize_unbound_partitioned` once every
   * rank has bound a buffer.
   *
   * @param storage_data The inline storage produced by the task.
   * @param domain The zero-based domain (extents) of the produced piece.
   */
  void bind(InternalSharedPtr<InlineStorage> storage_data, const Domain& domain);

  void finalize_unbound();

  /**
   * @brief Complete a multi-rank unbound binding once all ranks' pieces are known.
   *
   * Sets the global shape, re-bases this rank's local piece into its global sub-domain, records the
   * whole-store domain, and installs the given (Opaque) key partition.
   *
   * @param global_extents      The stitched-together global extents of the store.
   * @param local_global_domain This rank's piece expressed in global coordinates.
   * @param key_partition       The completed Opaque partition describing all ranks' pieces.
   */
  void finalize_unbound_partitioned(SmallVector<std::uint64_t, LEGATE_MAX_DIM> global_extents,
                                    const Domain& local_global_domain,
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
  Domain domain_{};
  std::uint32_t field_size_{};
  std::string_view provenance_{};  // only relevant on the root Storage

  InternalSharedPtr<InlineStorage> storage_data_{};

  std::optional<InternalSharedPtr<Partition>> key_partition_{};
};

}  // namespace legate::detail

#include <legate/data/detail/storage.inl>
