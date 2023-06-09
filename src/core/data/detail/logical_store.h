/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <optional>

#include "core/data/detail/logical_region_field.h"
#include "core/data/slice.h"
#include "core/data/store.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/restriction.h"

namespace legate::mapping {
class MachineDesc;
}  // namespace legate::mapping

namespace legate::detail {

class ProjectionInfo;
class StoragePartition;
class LogicalStorePartition;

class Storage : public std::enable_shared_from_this<Storage> {
 public:
  enum class Kind : int32_t {
    REGION_FIELD = 0,
    FUTURE       = 1,
  };

 public:
  // Create a RegionField-backed storage whose size is unbound. Initialized lazily.
  Storage(int32_t dim, std::unique_ptr<Type> type);
  // Create a RegionField-backed or a Future-backed storage. Initialized lazily.
  Storage(const Shape& extents, std::unique_ptr<Type> type, bool optimize_scalar);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(const Shape& extents, std::unique_ptr<Type> type, const Legion::Future& future);
  // Create a RegionField-bakced sub-storage. Initialized lazily.
  Storage(Shape&& extents,
          std::unique_ptr<Type> type,
          std::shared_ptr<StoragePartition> parent,
          Shape&& color,
          Shape&& offsets);

 public:
  bool unbound() const { return unbound_; }
  const Shape& extents() const;
  const Shape& offsets() const;
  size_t volume() const;
  int32_t dim() { return dim_; }
  const Type& type() const { return *type_; }
  Kind kind() const { return kind_; }
  int32_t level() const { return level_; }

 public:
  std::shared_ptr<Storage> slice(Shape tile_shape, Shape offsets);
  std::shared_ptr<const Storage> get_root() const;
  std::shared_ptr<Storage> get_root();

 public:
  LogicalRegionField* get_region_field();
  Legion::Future get_future() const;
  void set_region_field(std::shared_ptr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

 public:
  RegionField map(LibraryContext* context);

 public:
  Restrictions compute_restrictions() const;
  Partition* find_key_partition(const mapping::MachineDesc& machine,
                                const Restrictions& restrictions) const;
  void set_key_partition(const mapping::MachineDesc& machine,
                         std::unique_ptr<Partition>&& key_partition);
  void reset_key_partition();

 public:
  std::shared_ptr<StoragePartition> create_partition(std::shared_ptr<Partition> partition,
                                                     std::optional<bool> complete = std::nullopt);

 public:
  std::string to_string() const;

 private:
  uint64_t storage_id_{0};
  bool unbound_{false};
  int32_t dim_{-1};
  Shape extents_;
  size_t volume_;
  std::unique_ptr<Type> type_{nullptr};
  Kind kind_{Kind::REGION_FIELD};

 private:
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  Legion::Future future_{};

 private:
  int32_t level_{0};
  std::shared_ptr<StoragePartition> parent_{nullptr};
  Shape color_{};
  // Unlike offsets in a tiling, these offsets can never be negative, as a slicing always selects a
  // sub-rectangle of its parent
  Shape offsets_{};

 private:
  uint32_t num_pieces_{0};
  std::unique_ptr<Partition> key_partition_{nullptr};
};

class StoragePartition : public std::enable_shared_from_this<StoragePartition> {
 public:
  StoragePartition(std::shared_ptr<Storage> parent,
                   std::shared_ptr<Partition> partition,
                   bool complete);

 public:
  std::shared_ptr<Partition> partition() const { return partition_; }
  std::shared_ptr<const Storage> get_root() const;
  std::shared_ptr<Storage> get_root();
  std::shared_ptr<Storage> get_child_storage(const Shape& color);
  std::shared_ptr<LogicalRegionField> get_child_data(const Shape& color);

 public:
  Partition* find_key_partition(const mapping::MachineDesc& machine,
                                const Restrictions& restrictions) const;
  Legion::LogicalPartition get_legion_partition();

 public:
  int32_t level() const { return level_; }

 public:
  bool is_disjoint_for(const Domain* launch_domain) const;

 private:
  bool complete_;
  int32_t level_;
  std::shared_ptr<Storage> parent_;
  std::shared_ptr<Partition> partition_;
};

class LogicalStore : public std::enable_shared_from_this<LogicalStore> {
 public:
  LogicalStore(std::shared_ptr<Storage>&& storage);
  LogicalStore(Shape&& extents,
               const std::shared_ptr<Storage>& storage,
               std::shared_ptr<TransformStack>&& transform);

 public:
  ~LogicalStore();

 private:
  LogicalStore(std::shared_ptr<detail::LogicalStore> impl);

 public:
  LogicalStore(const LogicalStore& other)            = default;
  LogicalStore& operator=(const LogicalStore& other) = default;

 public:
  LogicalStore(LogicalStore&& other)            = default;
  LogicalStore& operator=(LogicalStore&& other) = default;

 public:
  bool unbound() const;
  const Shape& extents() const;
  size_t volume() const;
  // Size of the backing storage
  size_t storage_size() const;
  int32_t dim() const;
  bool has_scalar_storage() const;
  const Type& type() const;
  bool transformed() const;

 public:
  const Storage* get_storage() const;
  LogicalRegionField* get_region_field();
  Legion::Future get_future();
  void set_region_field(std::shared_ptr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

 public:
  std::shared_ptr<LogicalStore> promote(int32_t extra_dim, size_t dim_size);
  std::shared_ptr<LogicalStore> project(int32_t dim, int64_t index);
  std::shared_ptr<LogicalStore> slice(int32_t dim, Slice sl);
  std::shared_ptr<LogicalStore> transpose(std::vector<int32_t>&& axes);
  std::shared_ptr<LogicalStore> delinearize(int32_t dim, std::vector<int64_t>&& sizes);

 public:
  std::shared_ptr<LogicalStorePartition> partition_by_tiling(Shape tile_shape);

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  Restrictions compute_restrictions() const;
  std::shared_ptr<Partition> find_or_create_key_partition(const mapping::MachineDesc& machine,
                                                          const Restrictions& restrictions);
  bool has_key_partition(const mapping::MachineDesc& machine,
                         const Restrictions& restrictions) const;
  void set_key_partition(const mapping::MachineDesc& machine, const Partition* partition);
  void reset_key_partition();

 public:
  std::shared_ptr<LogicalStorePartition> create_partition(
    std::shared_ptr<Partition> partition, std::optional<bool> complete = std::nullopt);
  Legion::ProjectionID compute_projection(int32_t launch_ndim) const;

 public:
  void pack(BufferBuilder& buffer) const;

 public:
  std::string to_string() const;

 private:
  uint64_t store_id_;
  Shape extents_;
  std::shared_ptr<Storage> storage_;
  std::shared_ptr<TransformStack> transform_;

 private:
  uint32_t num_pieces_{0};
  std::shared_ptr<Partition> key_partition_{nullptr};
  std::shared_ptr<Store> mapped_{nullptr};
};

class LogicalStorePartition : public std::enable_shared_from_this<LogicalStorePartition> {
 public:
  LogicalStorePartition(std::shared_ptr<Partition> partition,
                        std::shared_ptr<StoragePartition> storage_partition,
                        std::shared_ptr<LogicalStore> store);

 public:
  std::shared_ptr<Partition> partition() const { return partition_; }
  std::shared_ptr<StoragePartition> storage_partition() const { return storage_partition_; }
  std::shared_ptr<LogicalStore> store() const { return store_; }
  std::unique_ptr<ProjectionInfo> create_projection_info(const Domain* launch_domain);
  bool is_disjoint_for(const Domain* launch_domain) const;

 private:
  std::shared_ptr<Partition> partition_;
  std::shared_ptr<StoragePartition> storage_partition_;
  std::shared_ptr<LogicalStore> store_;
};

}  // namespace legate::detail
