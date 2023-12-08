/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/detail/logical_region_field.h"
#include "core/data/detail/physical_store.h"
#include "core/data/physical_store.h"
#include "core/data/slice.h"
#include "core/mapping/detail/machine.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/projection.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/restriction.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/internal_shared_ptr.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace legate::detail {

class LogicalStorePartition;
struct ProjectionInfo;
class Strategy;
class StoragePartition;
class PhysicalStore;
class Variable;

class Storage : public legate::EnableSharedFromThis<Storage> {
 public:
  enum class Kind : int32_t {
    REGION_FIELD,
    FUTURE,
  };

  // Create a RegionField-backed storage whose size is unbound. Initialized lazily.
  Storage(int32_t dim, InternalSharedPtr<Type> type);
  // Create a RegionField-backed or a Future-backed storage. Initialized lazily.
  Storage(const Shape& extents, InternalSharedPtr<Type> type, bool optimize_scalar);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(const Shape& extents, InternalSharedPtr<Type> type, const Legion::Future& future);
  // Create a RegionField-backed sub-storage. Initialized lazily.
  Storage(Shape&& extents,
          InternalSharedPtr<Type> type,
          InternalSharedPtr<StoragePartition> parent,
          Shape&& color,
          Shape&& offsets);
  ~Storage();

  [[nodiscard]] uint64_t id() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const Shape& extents() const;
  [[nodiscard]] const Shape& offsets() const;
  [[nodiscard]] size_t volume() const;
  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<Storage>& other) const;
  [[nodiscard]] InternalSharedPtr<Type> type() const;
  [[nodiscard]] Kind kind() const;
  [[nodiscard]] int32_t level() const;

  [[nodiscard]] InternalSharedPtr<Storage> slice(Shape tile_shape, const Shape& offsets);
  [[nodiscard]] InternalSharedPtr<const Storage> get_root() const;
  [[nodiscard]] InternalSharedPtr<Storage> get_root();

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_region_field();
  [[nodiscard]] Legion::Future get_future() const;
  void set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

  [[nodiscard]] RegionField map();
  void allow_out_of_order_destruction();

  [[nodiscard]] Restrictions compute_restrictions() const;
  [[nodiscard]] Partition* find_key_partition(const mapping::detail::Machine& machine,
                                              const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         std::unique_ptr<Partition>&& key_partition);
  void reset_key_partition() noexcept;

  [[nodiscard]] InternalSharedPtr<StoragePartition> create_partition(
    InternalSharedPtr<Partition> partition, std::optional<bool> complete = std::nullopt);

  [[nodiscard]] std::string to_string() const;

 private:
  uint64_t storage_id_{};
  bool unbound_{};
  bool destroyed_out_of_order_{};
  int32_t dim_{-1};
  Shape extents_;
  InternalSharedPtr<Type> type_{};
  Kind kind_{Kind::REGION_FIELD};

  InternalSharedPtr<LogicalRegionField> region_field_{};
  std::unique_ptr<Legion::Future> future_{};

  int32_t level_{};
  InternalSharedPtr<StoragePartition> parent_{};
  Shape color_{};
  // Unlike offsets in a tiling, these offsets can never be negative, as a slicing always selects a
  // sub-rectangle of its parent
  Shape offsets_{};

  uint32_t num_pieces_{};
  std::unique_ptr<Partition> key_partition_{};
};

class StoragePartition : public legate::EnableSharedFromThis<StoragePartition> {
 public:
  StoragePartition(InternalSharedPtr<Storage> parent,
                   InternalSharedPtr<Partition> partition,
                   bool complete);

  [[nodiscard]] InternalSharedPtr<Partition> partition() const;
  [[nodiscard]] InternalSharedPtr<const Storage> get_root() const;
  [[nodiscard]] InternalSharedPtr<Storage> get_root();
  [[nodiscard]] InternalSharedPtr<Storage> get_child_storage(Shape color);
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_child_data(const Shape& color);

  [[nodiscard]] Partition* find_key_partition(const mapping::detail::Machine& machine,
                                              const Restrictions& restrictions) const;
  [[nodiscard]] Legion::LogicalPartition get_legion_partition();

  [[nodiscard]] int32_t level() const;

  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;

 private:
  bool complete_{};
  int32_t level_{};
  InternalSharedPtr<Storage> parent_{};
  InternalSharedPtr<Partition> partition_{};
};

class LogicalStore {
 public:
  explicit LogicalStore(InternalSharedPtr<Storage>&& storage);
  LogicalStore(Shape&& extents,
               InternalSharedPtr<Storage> storage,
               InternalSharedPtr<TransformStack>&& transform);

 private:
  explicit LogicalStore(InternalSharedPtr<detail::LogicalStore> impl);

 public:
  LogicalStore(LogicalStore&& other) noexcept            = default;
  LogicalStore& operator=(LogicalStore&& other) noexcept = default;

  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const Shape& extents() const;
  [[nodiscard]] size_t volume() const;
  // Size of the backing storage
  [[nodiscard]] size_t storage_size() const;
  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<LogicalStore>& other) const;
  [[nodiscard]] bool has_scalar_storage() const;
  [[nodiscard]] InternalSharedPtr<Type> type() const;
  [[nodiscard]] const InternalSharedPtr<TransformStack>& transform() const;
  [[nodiscard]] bool transformed() const;
  [[nodiscard]] uint64_t id() const;

  [[nodiscard]] const Storage* get_storage() const;
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_region_field();
  [[nodiscard]] Legion::Future get_future();
  void set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

  [[nodiscard]] InternalSharedPtr<LogicalStore> promote(int32_t extra_dim, size_t dim_size);
  [[nodiscard]] InternalSharedPtr<LogicalStore> project(int32_t dim, int64_t index);

 private:
  friend InternalSharedPtr<LogicalStore> slice_store(const InternalSharedPtr<LogicalStore>& self,
                                                     int32_t dim,
                                                     Slice sl);
  [[nodiscard]] InternalSharedPtr<LogicalStore> slice(const InternalSharedPtr<LogicalStore>& self,
                                                      int32_t dim,
                                                      Slice sl);

 public:
  [[nodiscard]] InternalSharedPtr<LogicalStore> transpose(const std::vector<int32_t>& axes);
  [[nodiscard]] InternalSharedPtr<LogicalStore> transpose(std::vector<int32_t>&& axes);
  [[nodiscard]] InternalSharedPtr<LogicalStore> delinearize(int32_t dim,
                                                            std::vector<uint64_t> sizes);

 private:
  friend InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
    const InternalSharedPtr<LogicalStore>& self, Shape tile_shape);
  [[nodiscard]] InternalSharedPtr<LogicalStorePartition> partition_by_tiling(
    const InternalSharedPtr<LogicalStore>& self, Shape tile_shape);

 public:
  [[nodiscard]] InternalSharedPtr<PhysicalStore> get_physical_store();
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

  [[nodiscard]] Restrictions compute_restrictions() const;
  [[nodiscard]] InternalSharedPtr<Partition> find_or_create_key_partition(
    const mapping::detail::Machine& machine, const Restrictions& restrictions);

  [[nodiscard]] InternalSharedPtr<Partition> get_current_key_partition() const;
  [[nodiscard]] bool has_key_partition(const mapping::detail::Machine& machine,
                                       const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine, const Partition* partition);
  void reset_key_partition();

 private:
  friend InternalSharedPtr<LogicalStorePartition> create_store_partition(
    const InternalSharedPtr<LogicalStore>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete);
  [[nodiscard]] InternalSharedPtr<LogicalStorePartition> create_partition(
    const InternalSharedPtr<LogicalStore>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete = std::nullopt);

 public:
  [[nodiscard]] Legion::ProjectionID compute_projection(
    int32_t launch_ndim, const std::optional<SymbolicPoint>& projection = {}) const;

  void pack(BufferBuilder& buffer) const;

 private:
  friend std::unique_ptr<Analyzable> store_to_launcher_arg(
    const InternalSharedPtr<LogicalStore>& self,
    const Variable* variable,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    int64_t redop);
  friend std::unique_ptr<Analyzable> store_to_launcher_arg_for_fixup(
    const InternalSharedPtr<LogicalStore>& self,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege);

  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg(
    const InternalSharedPtr<LogicalStore>& self,
    const Variable* variable,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    int64_t redop);
  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const InternalSharedPtr<LogicalStore>& self,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege);

 public:
  [[nodiscard]] std::string to_string() const;

 private:
  uint64_t store_id_{};
  Shape extents_{};
  InternalSharedPtr<Storage> storage_{};
  InternalSharedPtr<TransformStack> transform_{};

  uint32_t num_pieces_{};
  InternalSharedPtr<Partition> key_partition_{};
  InternalSharedPtr<PhysicalStore> mapped_{};
};

class LogicalStorePartition {
 public:
  LogicalStorePartition(InternalSharedPtr<Partition> partition,
                        InternalSharedPtr<StoragePartition> storage_partition,
                        InternalSharedPtr<LogicalStore> store);

  [[nodiscard]] InternalSharedPtr<Partition> partition() const;
  [[nodiscard]] InternalSharedPtr<StoragePartition> storage_partition() const;
  [[nodiscard]] InternalSharedPtr<LogicalStore> store() const;
  [[nodiscard]] InternalSharedPtr<LogicalStore> get_child_store(const Shape& color) const;
  [[nodiscard]] std::unique_ptr<ProjectionInfo> create_projection_info(
    const Domain& launch_domain, const std::optional<SymbolicPoint>& projection = {});
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;
  [[nodiscard]] const Shape& color_shape() const;

 private:
  InternalSharedPtr<Partition> partition_{};
  InternalSharedPtr<StoragePartition> storage_partition_{};
  InternalSharedPtr<LogicalStore> store_{};
};

[[nodiscard]] InternalSharedPtr<LogicalStore> slice_store(
  const InternalSharedPtr<LogicalStore>& self, int32_t dim, Slice sl);

[[nodiscard]] InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
  const InternalSharedPtr<LogicalStore>& self, Shape tile_shape);

[[nodiscard]] InternalSharedPtr<LogicalStorePartition> create_store_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete = std::nullopt);

[[nodiscard]] std::unique_ptr<Analyzable> store_to_launcher_arg(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  int64_t redop = -1);

[[nodiscard]] std::unique_ptr<Analyzable> store_to_launcher_arg_for_fixup(
  const InternalSharedPtr<LogicalStore>& self,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege);

}  // namespace legate::detail

#include "core/data/detail/logical_store.inl"
