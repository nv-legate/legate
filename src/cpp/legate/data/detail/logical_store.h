/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/storage.h>
#include <legate/data/detail/storage_partition.h>
#include <legate/data/slice.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/task/detail/task_return_layout.h>
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

class LogicalStorePartition;
class TransformStack;
class PhysicalStore;
class TaskReturnLayoutForUnpack;
class Strategy;
class RegionPhysicalStore;

class LogicalStore {
 public:
  LogicalStore(InternalSharedPtr<Storage> storage, InternalSharedPtr<Type> type);
  // This constructor is invoked exclusively by store transformations that construct stores from
  // immediate extents.
  LogicalStore(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
               InternalSharedPtr<Storage> storage,
               InternalSharedPtr<Type> type,
               InternalSharedPtr<TransformStack> transform);

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
  [[nodiscard]] bool has_scalar_storage() const;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;
  [[nodiscard]] const InternalSharedPtr<TransformStack>& transform() const;
  [[nodiscard]] bool transformed() const;
  [[nodiscard]] std::uint64_t id() const;

  [[nodiscard]] const InternalSharedPtr<Storage>& get_storage() const;
  [[nodiscard]] const InternalSharedPtr<LogicalRegionField>& get_region_field() const;
  [[nodiscard]] Legion::Future get_future() const;
  [[nodiscard]] Legion::FutureMap get_future_map() const;
  void set_region_field(InternalSharedPtr<LogicalRegionField> region_field);
  void set_future(Legion::Future future, std::size_t scalar_offset = 0);
  void set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset = 0);

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
    std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes);

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
  [[nodiscard]] InternalSharedPtr<PhysicalStore> get_physical_store(
    legate::mapping::StoreTarget target, bool ignore_future_mutability);

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
  // Informs the runtime that references to this store may be removed in non-deterministic order
  // (e.g. by an asynchronous garbage collector).
  //
  // Normally the top-level code must perform all Legate operations in a deterministic order (at
  // least when running on multiple ranks/nodes). This includes destruction of objects managing
  // Legate state, like stores. Passing a reference to such an object to a garbage-collected
  // language violates this assumption, because (in general) garbage collection can occur at
  // indeterminate points during the execution, and thus the point when the object's reference count
  // drops to 0 (which triggers object destruction) is not deterministic.
  //
  // Before passing a store to a garbage collected language, it must first be marked using this
  // function, so that the runtime knows to work around the potentially non-deterministic removal of
  // references.
  void allow_out_of_order_destruction();

  [[nodiscard]] Restrictions compute_restrictions(bool is_output) const;
  [[nodiscard]] InternalSharedPtr<Partition> find_or_create_key_partition(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions);

  /**
   * @brief Gets the current key partition of this store if it exists. To get the most up-to-date
   * partition, the user must call `Runtime::flush_scheduling_window()` before calling this
   * function.
   *
   * @return The current key partition of this store.
   */
  [[nodiscard]] const std::optional<InternalSharedPtr<Partition>>& get_current_key_partition()
    const;

  [[nodiscard]] bool has_key_partition(const mapping::detail::Machine& machine,
                                       const ParallelPolicy& parallel_policy,
                                       const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         const ParallelPolicy& parallel_policy,
                         InternalSharedPtr<Partition> partition);
  void reset_key_partition();

 private:
  /**
   * @brief Immediately reset the key partition of this store if it matches with `to_match`.
   *
   * Unlike the public `reset_key_partition` method, this method doesn't flush the scheduling
   * window, as it is meant to be used exclusively by invalidation callbacks, which are invoked
   * during scheduling.
   */
  void maybe_reset_key_partition_(const Partition* to_match) noexcept;

  friend InternalSharedPtr<LogicalStorePartition> create_store_partition(
    const InternalSharedPtr<LogicalStore>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete);
  [[nodiscard]] InternalSharedPtr<LogicalStorePartition> create_partition_(
    const InternalSharedPtr<LogicalStore>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete = std::nullopt);

 public:
  [[nodiscard]] Legion::ProjectionID compute_projection(
    const Domain& launch_domain,
    Span<const std::uint64_t> color_shape,
    const std::optional<SymbolicPoint>& projection = {}) const;

  void pack(BufferBuilder& buffer) const;
  void calculate_pack_size(TaskReturnLayoutForUnpack* layout) const;

 private:
  friend StoreAnalyzable store_to_launcher_arg(const InternalSharedPtr<LogicalStore>& self,
                                               const Variable* variable,
                                               const Strategy& strategy,
                                               const Domain& launch_domain,
                                               const std::optional<SymbolicPoint>& projection,
                                               Legion::PrivilegeMode privilege,
                                               GlobalRedopID redop);
  friend RegionFieldArg store_to_launcher_arg_for_fixup(const InternalSharedPtr<LogicalStore>& self,
                                                        const Domain& launch_domain,
                                                        Legion::PrivilegeMode privilege);

  [[nodiscard]] StoreAnalyzable to_launcher_arg_(const InternalSharedPtr<LogicalStore>& self,
                                                 const Variable* variable,
                                                 const Strategy& strategy,
                                                 const Domain& launch_domain,
                                                 const std::optional<SymbolicPoint>& projection,
                                                 Legion::PrivilegeMode privilege,
                                                 GlobalRedopID redop);
  [[nodiscard]] RegionFieldArg to_launcher_arg_for_fixup_(
    const InternalSharedPtr<LogicalStore>& self,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege);

  [[nodiscard]] StoreAnalyzable future_to_launcher_arg_(Legion::Future future,
                                                        const Domain& launch_domain,
                                                        Legion::PrivilegeMode privilege,
                                                        GlobalRedopID redop);
  [[nodiscard]] StoreAnalyzable future_map_to_launcher_arg_(const Domain& launch_domain,
                                                            Legion::PrivilegeMode privilege,
                                                            GlobalRedopID redop);
  [[nodiscard]] StoreAnalyzable region_field_to_launcher_arg_(
    const InternalSharedPtr<LogicalStore>& self,
    const Variable* variable,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    GlobalRedopID redop);

 public:
  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] bool equal_storage(const LogicalStore& other) const;

 private:
  std::uint64_t store_id_{};
  InternalSharedPtr<Type> type_{};
  InternalSharedPtr<Shape> shape_{};
  InternalSharedPtr<Storage> storage_{};
  InternalSharedPtr<TransformStack> transform_{};

  std::uint32_t num_pieces_{};
  std::optional<InternalSharedPtr<Partition>> key_partition_{};
  std::optional<InternalSharedPtr<PhysicalStore>> mapped_{};
};

[[nodiscard]] InternalSharedPtr<LogicalStore> slice_store(
  const InternalSharedPtr<LogicalStore>& self, std::int32_t dim, Slice sl);

[[nodiscard]] StoreAnalyzable store_to_launcher_arg(const InternalSharedPtr<LogicalStore>& self,
                                                    const Variable* variable,
                                                    const Strategy& strategy,
                                                    const Domain& launch_domain,
                                                    const std::optional<SymbolicPoint>& projection,
                                                    Legion::PrivilegeMode privilege,
                                                    GlobalRedopID redop = GlobalRedopID{-1});

[[nodiscard]] RegionFieldArg store_to_launcher_arg_for_fixup(
  const InternalSharedPtr<LogicalStore>& self,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege);

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
