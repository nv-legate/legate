/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/physical_store.h>
#include <legate/data/slice.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <optional>
#include <string>
#include <variant>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::detail {

class LogicalStorePartition;
class PhysicalStore;
class StoragePartition;
class Strategy;
class TaskReturnLayoutForUnpack;
class Variable;

class Storage {
 public:
  enum class Kind : std::uint8_t {
    REGION_FIELD,
    FUTURE,
    FUTURE_MAP,
  };

  // Create a RegionField-backed or a Future-backed storage.
  Storage(InternalSharedPtr<Shape> shape,
          std::uint32_t field_size,
          bool optimize_scalar,
          std::string_view provenance);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(InternalSharedPtr<Shape> shape, Legion::Future future, std::string_view provenance);
  // Create a RegionField-backed sub-storage.
  Storage(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
          InternalSharedPtr<StoragePartition> parent,
          SmallVector<std::uint64_t, LEGATE_MAX_DIM> color,
          SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);
  ~Storage();

  Storage(Storage&&) noexcept            = default;
  Storage& operator=(Storage&&) noexcept = default;

  [[nodiscard]] std::uint64_t id() const;
  [[nodiscard]] bool replicated() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const;
  [[nodiscard]] Span<const std::uint64_t> extents() const;
  [[nodiscard]] Span<const std::int64_t> offsets() const;
  [[nodiscard]] std::size_t volume() const;
  [[nodiscard]] std::uint32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<Storage>& other) const;
  [[nodiscard]] Kind kind() const;
  [[nodiscard]] std::int32_t level() const;
  [[nodiscard]] std::size_t scalar_offset() const;
  [[nodiscard]] std::string_view provenance() const;
  [[nodiscard]] bool is_mapped() const;

  [[nodiscard]] InternalSharedPtr<Storage> slice(
    const InternalSharedPtr<Storage>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
    SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);
  [[nodiscard]] const Storage* get_root() const;
  [[nodiscard]] Storage* get_root();
  [[nodiscard]] InternalSharedPtr<const Storage> get_root(
    const InternalSharedPtr<const Storage>& self) const;
  [[nodiscard]] InternalSharedPtr<Storage> get_root(const InternalSharedPtr<Storage>& self);

  [[nodiscard]] const InternalSharedPtr<LogicalRegionField>& get_region_field() const noexcept;
  [[nodiscard]] Legion::Future get_future() const;
  [[nodiscard]] Legion::FutureMap get_future_map() const;
  [[nodiscard]] std::variant<Legion::Future, Legion::FutureMap> get_future_or_future_map(
    const Domain& launch_domain) const;

  void set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future, std::size_t scalar_offset);
  void set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset);

  [[nodiscard]] RegionField map(legate::mapping::StoreTarget target);
  void unmap();
  void allow_out_of_order_destruction();
  void free_early() noexcept;

  [[nodiscard]] Restrictions compute_restrictions() const;
  [[nodiscard]] std::optional<InternalSharedPtr<Partition>> find_key_partition(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         InternalSharedPtr<Partition> key_partition);
  void reset_key_partition() noexcept;

  [[nodiscard]] InternalSharedPtr<StoragePartition> create_partition(
    const InternalSharedPtr<Storage>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete = std::nullopt);

  [[nodiscard]] std::string to_string() const;

 private:
  std::uint64_t storage_id_{};
  bool replicated_{};
  bool unbound_{};
  bool destroyed_out_of_order_{};  // only relevant on the root Storage
  InternalSharedPtr<Shape> shape_{};
  Kind kind_{Kind::REGION_FIELD};
  std::string_view provenance_{};  // only relevant on the root Storage

  std::int32_t level_{};
  std::optional<InternalSharedPtr<StoragePartition>> parent_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_{};
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets_{};

  std::size_t scalar_offset_{};
  std::optional<InternalSharedPtr<LogicalRegionField>> region_field_{};
  std::optional<Legion::Future> future_{};
  std::optional<Legion::FutureMap> future_map_{};

  std::uint32_t num_pieces_{};
  std::optional<InternalSharedPtr<Partition>> key_partition_{};
};

class StoragePartition {
 public:
  StoragePartition(InternalSharedPtr<Storage> parent,
                   InternalSharedPtr<Partition> partition,
                   bool complete);

  [[nodiscard]] const InternalSharedPtr<Partition>& partition() const;
  [[nodiscard]] const Storage* get_root() const;
  [[nodiscard]] Storage* get_root();
  [[nodiscard]] InternalSharedPtr<const Storage> get_root(
    const InternalSharedPtr<const StoragePartition>&) const;
  [[nodiscard]] InternalSharedPtr<Storage> get_root(const InternalSharedPtr<StoragePartition>&);
  [[nodiscard]] InternalSharedPtr<Storage> get_child_storage(
    const InternalSharedPtr<StoragePartition>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color);
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_child_data(
    Span<const std::uint64_t> color);

  [[nodiscard]] std::optional<InternalSharedPtr<Partition>> find_key_partition(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions) const;
  [[nodiscard]] Legion::LogicalPartition get_legion_partition();

  [[nodiscard]] std::int32_t level() const;

  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;

 private:
  bool complete_{};
  std::int32_t level_{};
  InternalSharedPtr<Storage> parent_{};
  InternalSharedPtr<Partition> partition_{};
};

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
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape);
  [[nodiscard]] InternalSharedPtr<LogicalStorePartition> partition_by_tiling_(
    const InternalSharedPtr<LogicalStore>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape);

 public:
  [[nodiscard]] InternalSharedPtr<PhysicalStore> get_physical_store(
    legate::mapping::StoreTarget target, bool ignore_future_mutability);
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

  [[nodiscard]] const InternalSharedPtr<Partition>& get_current_key_partition() const;
  [[nodiscard]] bool has_key_partition(const mapping::detail::Machine& machine,
                                       const ParallelPolicy& parallel_policy,
                                       const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         const ParallelPolicy& parallel_policy,
                         InternalSharedPtr<Partition> partition);
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
  InternalSharedPtr<PhysicalStore> mapped_{};
};

class LogicalStorePartition {
 public:
  LogicalStorePartition(InternalSharedPtr<Partition> partition,
                        InternalSharedPtr<StoragePartition> storage_partition,
                        InternalSharedPtr<LogicalStore> store);

  [[nodiscard]] const InternalSharedPtr<Partition>& partition() const;
  [[nodiscard]] const InternalSharedPtr<StoragePartition>& storage_partition() const;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& store() const;
  [[nodiscard]] InternalSharedPtr<LogicalStore> get_child_store(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const;
  [[nodiscard]] StoreProjection create_store_projection(
    const Domain& launch_domain, const std::optional<SymbolicPoint>& projection = {});
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;
  [[nodiscard]] Span<const std::uint64_t> color_shape() const;

 private:
  InternalSharedPtr<Partition> partition_{};
  InternalSharedPtr<StoragePartition> storage_partition_{};
  InternalSharedPtr<LogicalStore> store_{};
};

[[nodiscard]] InternalSharedPtr<StoragePartition> create_storage_partition(
  const InternalSharedPtr<Storage>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete);

[[nodiscard]] InternalSharedPtr<Storage> slice_storage(
  const InternalSharedPtr<Storage>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);

[[nodiscard]] InternalSharedPtr<LogicalStore> slice_store(
  const InternalSharedPtr<LogicalStore>& self, std::int32_t dim, Slice sl);

[[nodiscard]] InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
  const InternalSharedPtr<LogicalStore>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape);

[[nodiscard]] InternalSharedPtr<LogicalStorePartition> create_store_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete = std::nullopt);

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

}  // namespace legate::detail

#include <legate/data/detail/logical_store.inl>
