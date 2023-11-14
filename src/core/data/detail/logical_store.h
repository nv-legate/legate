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
#include "core/partitioning/partition.h"
#include "core/partitioning/restriction.h"
#include "core/runtime/detail/projection.h"
#include "core/utilities/detail/buffer_builder.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace legate::detail {

struct Analyzable;
class LogicalStorePartition;
struct ProjectionInfo;
class Strategy;
class StoragePartition;
class PhysicalStore;
class Variable;

class Storage : public std::enable_shared_from_this<Storage> {
 public:
  enum class Kind : int32_t {
    REGION_FIELD,
    FUTURE,
  };

  // Create a RegionField-backed storage whose size is unbound. Initialized lazily.
  Storage(int32_t dim, std::shared_ptr<Type> type);
  // Create a RegionField-backed or a Future-backed storage. Initialized lazily.
  Storage(const Shape& extents, std::shared_ptr<Type> type, bool optimize_scalar);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(const Shape& extents, std::shared_ptr<Type> type, const Legion::Future& future);
  // Create a RegionField-backed sub-storage. Initialized lazily.
  Storage(Shape&& extents,
          std::shared_ptr<Type> type,
          std::shared_ptr<StoragePartition> parent,
          Shape&& color,
          Shape&& offsets);
  ~Storage();

  [[nodiscard]] uint64_t id() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const Shape& extents() const;
  [[nodiscard]] const Shape& offsets() const;
  [[nodiscard]] size_t volume() const;
  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] bool overlaps(const std::shared_ptr<Storage>& other) const;
  [[nodiscard]] std::shared_ptr<Type> type() const;
  [[nodiscard]] Kind kind() const;
  [[nodiscard]] int32_t level() const;

  [[nodiscard]] std::shared_ptr<Storage> slice(Shape tile_shape, const Shape& offsets);
  [[nodiscard]] std::shared_ptr<const Storage> get_root() const;
  [[nodiscard]] std::shared_ptr<Storage> get_root();

  [[nodiscard]] std::shared_ptr<LogicalRegionField> get_region_field();
  [[nodiscard]] Legion::Future get_future() const;
  void set_region_field(std::shared_ptr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

  [[nodiscard]] RegionField map();
  void allow_out_of_order_destruction();

  [[nodiscard]] Restrictions compute_restrictions() const;
  [[nodiscard]] Partition* find_key_partition(const mapping::detail::Machine& machine,
                                              const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         std::unique_ptr<Partition>&& key_partition);
  void reset_key_partition() noexcept;

  [[nodiscard]] std::shared_ptr<StoragePartition> create_partition(
    std::shared_ptr<Partition> partition, std::optional<bool> complete = std::nullopt);

  [[nodiscard]] std::string to_string() const;

 private:
  uint64_t storage_id_{};
  bool unbound_{};
  bool destroyed_out_of_order_{};
  int32_t dim_{-1};
  Shape extents_;
  std::shared_ptr<Type> type_{};
  Kind kind_{Kind::REGION_FIELD};

  std::shared_ptr<LogicalRegionField> region_field_{};
  std::unique_ptr<Legion::Future> future_{};

  int32_t level_{};
  std::shared_ptr<StoragePartition> parent_{};
  Shape color_{};
  // Unlike offsets in a tiling, these offsets can never be negative, as a slicing always selects a
  // sub-rectangle of its parent
  Shape offsets_{};

  uint32_t num_pieces_{};
  std::unique_ptr<Partition> key_partition_{};
};

class StoragePartition : public std::enable_shared_from_this<StoragePartition> {
 public:
  StoragePartition(std::shared_ptr<Storage> parent,
                   std::shared_ptr<Partition> partition,
                   bool complete);

  [[nodiscard]] std::shared_ptr<Partition> partition() const;
  [[nodiscard]] std::shared_ptr<const Storage> get_root() const;
  [[nodiscard]] std::shared_ptr<Storage> get_root();
  [[nodiscard]] std::shared_ptr<Storage> get_child_storage(Shape color);
  [[nodiscard]] std::shared_ptr<LogicalRegionField> get_child_data(const Shape& color);

  [[nodiscard]] Partition* find_key_partition(const mapping::detail::Machine& machine,
                                              const Restrictions& restrictions) const;
  [[nodiscard]] Legion::LogicalPartition get_legion_partition();

  [[nodiscard]] int32_t level() const;

  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;

 private:
  bool complete_{};
  int32_t level_{};
  std::shared_ptr<Storage> parent_{};
  std::shared_ptr<Partition> partition_{};
};

class LogicalStore : public std::enable_shared_from_this<LogicalStore> {
 public:
  explicit LogicalStore(std::shared_ptr<Storage>&& storage);
  LogicalStore(Shape&& extents,
               const std::shared_ptr<Storage>& storage,
               std::shared_ptr<TransformStack>&& transform);

 private:
  explicit LogicalStore(std::shared_ptr<detail::LogicalStore> impl);

 public:
  LogicalStore(LogicalStore&& other) noexcept            = default;
  LogicalStore& operator=(LogicalStore&& other) noexcept = default;

  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const Shape& extents() const;
  [[nodiscard]] size_t volume() const;
  // Size of the backing storage
  [[nodiscard]] size_t storage_size() const;
  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] bool overlaps(const std::shared_ptr<LogicalStore>& other) const;
  [[nodiscard]] bool has_scalar_storage() const;
  [[nodiscard]] std::shared_ptr<Type> type() const;
  [[nodiscard]] bool transformed() const;
  [[nodiscard]] uint64_t id() const;

  [[nodiscard]] const Storage* get_storage() const;
  [[nodiscard]] std::shared_ptr<LogicalRegionField> get_region_field();
  [[nodiscard]] Legion::Future get_future();
  void set_region_field(std::shared_ptr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

  [[nodiscard]] std::shared_ptr<LogicalStore> promote(int32_t extra_dim, size_t dim_size);
  [[nodiscard]] std::shared_ptr<LogicalStore> project(int32_t dim, int64_t index);
  [[nodiscard]] std::shared_ptr<LogicalStore> slice(int32_t dim, Slice sl);
  [[nodiscard]] std::shared_ptr<LogicalStore> transpose(const std::vector<int32_t>& axes);
  [[nodiscard]] std::shared_ptr<LogicalStore> transpose(std::vector<int32_t>&& axes);
  [[nodiscard]] std::shared_ptr<LogicalStore> delinearize(int32_t dim,
                                                          const std::vector<int64_t>& sizes);
  [[nodiscard]] std::shared_ptr<LogicalStore> delinearize(int32_t dim,
                                                          std::vector<int64_t>&& sizes);

  [[nodiscard]] std::shared_ptr<LogicalStorePartition> partition_by_tiling(Shape tile_shape);

  [[nodiscard]] std::shared_ptr<PhysicalStore> get_physical_store();
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
  [[nodiscard]] std::shared_ptr<Partition> find_or_create_key_partition(
    const mapping::detail::Machine& machine, const Restrictions& restrictions);

  [[nodiscard]] std::shared_ptr<Partition> get_current_key_partition() const;
  [[nodiscard]] bool has_key_partition(const mapping::detail::Machine& machine,
                                       const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine, const Partition* partition);
  void reset_key_partition();

  [[nodiscard]] std::shared_ptr<LogicalStorePartition> create_partition(
    std::shared_ptr<Partition> partition, std::optional<bool> complete = std::nullopt);
  [[nodiscard]] Legion::ProjectionID compute_projection(
    int32_t launch_ndim, std::optional<proj::SymbolicFunctor> proj_fn = std::nullopt) const;

  void pack(BufferBuilder& buffer) const;
  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg(const Variable* variable,
                                                            const Strategy& strategy,
                                                            const Domain& launch_domain,
                                                            Legion::PrivilegeMode privilege,
                                                            int64_t redop = -1);
  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege);

  [[nodiscard]] std::string to_string() const;

 private:
  uint64_t store_id_{};
  Shape extents_{};
  std::shared_ptr<Storage> storage_{};
  std::shared_ptr<TransformStack> transform_{};

  uint32_t num_pieces_{};
  std::shared_ptr<Partition> key_partition_{};
  std::shared_ptr<PhysicalStore> mapped_{};
};

class LogicalStorePartition : public std::enable_shared_from_this<LogicalStorePartition> {
 public:
  LogicalStorePartition(std::shared_ptr<Partition> partition,
                        std::shared_ptr<StoragePartition> storage_partition,
                        std::shared_ptr<LogicalStore> store);

  [[nodiscard]] std::shared_ptr<Partition> partition() const;
  [[nodiscard]] std::shared_ptr<StoragePartition> storage_partition() const;
  [[nodiscard]] std::shared_ptr<LogicalStore> store() const;
  [[nodiscard]] std::unique_ptr<ProjectionInfo> create_projection_info(
    const Domain& launch_domain, std::optional<proj::SymbolicFunctor> proj_fn = std::nullopt);
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;
  [[nodiscard]] const Shape& color_shape() const;

 private:
  std::shared_ptr<Partition> partition_{};
  std::shared_ptr<StoragePartition> storage_partition_{};
  std::shared_ptr<LogicalStore> store_{};
};

}  // namespace legate::detail

#include "core/data/detail/logical_store.inl"
