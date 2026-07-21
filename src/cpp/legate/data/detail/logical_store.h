/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/storage.h>
#include <legate/data/slice.h>
#include <legate/mapping/mapping.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <optional>
#include <string>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class DataMovementRequests;
class LogicalStorePartition;
class TransformStack;
class Partition;
class PhysicalStore;

class LogicalStore {
 public:
  LogicalStore(InternalSharedPtr<Storage> storage, InternalSharedPtr<Type> type);
  /**
   * @brief Constructs a LogicalStore from a set of extents, storage, type, and transform.
   *
   * @note This constructor is invoked exclusively by the store transformations
   * that constructs stores from immediate extents. As a result, this constructor
   * should only be used for creating transformed stores.
   *
   * @param extents The extents representing the shape of the store.
   * @param storage The storage backing the store.
   * @param type The type of the LogicalStore.
   * @param transform The transform stack over the storage that produced this store.
   * @param domain The (sub)domain over the root storage that backs this store. This is used for
   * tracking domain information when avoiding sub-storages such as when inline task launching.
   *
   * @return The constructed LogicalStore.
   */
  LogicalStore(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
               InternalSharedPtr<Storage> storage,
               InternalSharedPtr<Type> type,
               InternalSharedPtr<TransformStack> transform,
               Domain domain);

  LogicalStore(LogicalStore&& other) noexcept            = default;
  LogicalStore& operator=(LogicalStore&& other) noexcept = default;

  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const;
  [[nodiscard]] Span<const std::uint64_t> extents() const;
  [[nodiscard]] std::size_t volume() const;
  // Size of the backing storage
  [[nodiscard]] std::size_t storage_size() const;
  [[nodiscard]] std::uint32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<LogicalStore>& other) const;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;
  [[nodiscard]] const InternalSharedPtr<TransformStack>& transform() const;
  [[nodiscard]] bool transformed() const;
  [[nodiscard]] std::uint64_t id() const;

  [[nodiscard]] const InternalSharedPtr<Storage>& get_storage() const;

  [[nodiscard]] InternalSharedPtr<LogicalStore> reinterpret_as(InternalSharedPtr<Type> type) const;

  [[nodiscard]] InternalSharedPtr<LogicalStore> promote(std::int32_t extra_dim,
                                                        std::size_t dim_size);
  [[nodiscard]] InternalSharedPtr<LogicalStore> project(std::int32_t dim, std::int64_t index);
  /**
   * @brief Return a view to the store where the unit-size dimension `dim` is broadcasted to a
   * dimension of size `dim_size`.
   */
  [[nodiscard]] InternalSharedPtr<LogicalStore> broadcast(std::int32_t dim, std::size_t dim_size);

 private:
  friend InternalSharedPtr<LogicalStore> slice_store(const InternalSharedPtr<LogicalStore>& self,
                                                     std::int32_t dim,
                                                     Slice sl);
  [[nodiscard]] InternalSharedPtr<LogicalStore> slice_(const InternalSharedPtr<LogicalStore>& self,
                                                       std::int32_t dim,
                                                       Slice sl);

 public:
  [[nodiscard]] InternalSharedPtr<LogicalStore> transpose(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> axes);
  [[nodiscard]] InternalSharedPtr<LogicalStore> delinearize(
    std::int32_t dim, const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& sizes);

 private:
  friend InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
    const InternalSharedPtr<LogicalStore>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
    std::optional<SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape);
  [[nodiscard]] InternalSharedPtr<LogicalStorePartition> partition_by_tiling_(
    const InternalSharedPtr<LogicalStore>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
    std::optional<SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape);

 public:
  [[nodiscard]] std::pair<InternalSharedPtr<PhysicalStore>, bool> map(
    const InternalSharedPtr<Partition>& partition,
    const SymbolicPoint& store_projection,
    const Domain& launch_domain,
    const DomainPoint& task_index,
    bool is_read,
    const legate::mapping::detail::Machine& machine,
    DataMovementRequests* requests);
  [[nodiscard]] InternalSharedPtr<PhysicalStore> map_reduction(
    const InternalSharedPtr<Partition>& partition,
    const SymbolicPoint& store_projection,
    const Domain& launch_domain,
    const DomainPoint& task_index,
    ReductionOpKind op_kind,
    bool store_has_identity,
    const legate::mapping::detail::Machine& machine,
    DataMovementRequests* requests);
  void reshuffle(const InternalSharedPtr<Partition>& new_key_partition);

  /**
   * @brief Maps an unbound store to a physical store with no backing storage.
   *
   * The returned physical store carries the memory kind a produced buffer should be allocated on
   * and a reference back to this logical store, so that a task binding a buffer can update this
   * store in place (see `bind_buffer`).
   *
   * @param self A shared pointer to this store (used as the originating-store back-reference).
   * @param target The memory kind produced buffers should be allocated on.
   *
   * @return The physical store handed to the task body.
   */
  [[nodiscard]] InternalSharedPtr<PhysicalStore> map_unbound(
    const InternalSharedPtr<LogicalStore>& self, legate::mapping::StoreTarget target);

  /**
   * @brief Binds a task-produced inline storage to this (unbound) store.
   *
   * Updates the store's shape/domain and backing storage in place. After this call the store is
   * no longer unbound and downstream operations observe the produced data.
   *
   * @param storage The inline storage produced by the task.
   * @param domain The domain the store should span over @p storage.
   */
  void bind_buffer(InternalSharedPtr<InlineStorage> storage, const Domain& domain);

  void finalize_unbound();

  /**
   * @brief Finalize a multi-rank unbound store after every rank has bound its piece.
   *
   * All-gathers each rank's produced piece (its color and local extents), stitches the pieces into
   * a single global store via dimension-wise prefix sums, fills the placeholder @p partition (an
   * `Opaque`) with the per-color global sub-domains, sets the store's global shape/domain, and
   * installs the completed partition as the store's key partition. Must be called collectively by
   * every rank participating in the producing task's launch.
   *
   * @param partition        The placeholder `Opaque` partition assigned during partitioning.
   * @param store_projection The store's projection (maps the launch point to its color).
   * @param task_index       This rank's launch point.
   */
  void finalize_unbound_partitioned(const InternalSharedPtr<Partition>& partition,
                                    const SymbolicPoint& store_projection,
                                    const DomainPoint& task_index);

  [[nodiscard]] InternalSharedPtr<PhysicalStore> get_physical_store(
    std::optional<legate::mapping::StoreTarget> target, bool ignore_future_mutability);

  /**
   * @brief Gets the currently mapped `PhysicalStore` for this `LogicalStore`
   *
   * Returns the physical store if this logical store has already been mapped to physical
   * memory via a previous call to `get_physical_store()`. This method does not trigger
   * any new mapping operations and does not block.
   *
   * This is useful for checking if a store has an existing physical allocation without
   * forcing a new inline mapping operation.
   */
  [[nodiscard]] const std::optional<InternalSharedPtr<PhysicalStore>>& get_cached_physical_store()
    const;
  [[nodiscard]] bool is_mapped() const;

  [[nodiscard]] bool needs_flush() const;
  void detach();

  [[nodiscard]] Restrictions compute_restrictions(bool is_output) const;
  [[nodiscard]] InternalSharedPtr<Partition> find_or_create_key_partition(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions);

  [[nodiscard]] const std::optional<InternalSharedPtr<Partition>>& get_current_key_partition()
    const;

  [[nodiscard]] bool has_key_partition(const mapping::detail::Machine& machine,
                                       const ParallelPolicy& parallel_policy,
                                       const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         const ParallelPolicy& parallel_policy,
                         InternalSharedPtr<Partition> partition);
  void set_key_partition(InternalSharedPtr<Partition> partition);
  void reset_key_partition();

 private:
  friend InternalSharedPtr<LogicalStorePartition> create_store_partition(
    const InternalSharedPtr<LogicalStore>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete);
  [[nodiscard]] InternalSharedPtr<LogicalStorePartition> create_partition_(
    const InternalSharedPtr<LogicalStore>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete = std::nullopt);

 public:
  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] bool equal_storage(const LogicalStore& other) const;

 private:
  std::uint64_t store_id_{};
  InternalSharedPtr<Type> type_{};
  InternalSharedPtr<Shape> shape_{};
  InternalSharedPtr<Storage> storage_{};
  InternalSharedPtr<TransformStack> transform_{};

  Domain domain_{};

  std::uint32_t num_pieces_{};
  std::optional<InternalSharedPtr<Partition>> key_partition_{};
  std::optional<InternalSharedPtr<PhysicalStore>> mapped_{};
};

[[nodiscard]] InternalSharedPtr<LogicalStore> slice_store(
  const InternalSharedPtr<LogicalStore>& self, std::int32_t dim, Slice sl);

[[nodiscard]] InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
  const InternalSharedPtr<LogicalStore>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  std::optional<SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape = std::nullopt);

[[nodiscard]] InternalSharedPtr<LogicalStorePartition> create_store_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete = std::nullopt);

}  // namespace legate::detail

#include <legate/data/detail/logical_store.inl>
