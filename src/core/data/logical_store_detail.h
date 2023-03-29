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

#include "core/data/logical_region_field.h"
#include "core/partitioning/partition.h"
#include "core/runtime/runtime.h"

namespace legate {
namespace detail {

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
  Storage(int32_t dim, LegateTypeCode code);
  // Create a RegionField-backed or a Future-backedstorage. Initialized lazily.
  Storage(Shape extents, LegateTypeCode code, bool optimize_scalar);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(Shape extents, LegateTypeCode code, const Legion::Future& future);

 public:
  bool unbound() const { return unbound_; }
  const Shape& extents() const { return extents_; }
  size_t volume() const { return volume_; }
  int32_t dim();
  LegateTypeCode code() const { return code_; }
  Kind kind() const { return kind_; }

 public:
  LogicalRegionField* get_region_field();
  Legion::Future get_future() const;
  void set_region_field(std::shared_ptr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

 public:
  RegionField map(LibraryContext* context);

 public:
  Partition* find_or_create_key_partition();
  void set_key_partition(std::unique_ptr<Partition>&& key_partition);
  void reset_key_partition();
  Legion::LogicalPartition find_or_create_legion_partition(const Partition* partition);

 public:
  std::shared_ptr<StoragePartition> create_partition(std::shared_ptr<Partition> partition);

 private:
  bool unbound_{false};
  int32_t dim_{-1};
  Shape extents_;
  size_t volume_;
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  Kind kind_{Kind::REGION_FIELD};
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  Legion::Future future_{};

 private:
  std::unique_ptr<Partition> key_partition_{nullptr};
};

class StoragePartition {
 public:
  StoragePartition(std::shared_ptr<Storage> parent, std::shared_ptr<Partition> partition);

 public:
  std::shared_ptr<Partition> partition() const { return partition_; }

 private:
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
  bool scalar() const;
  LegateTypeCode code() const;

 public:
  LogicalRegionField* get_region_field();
  Legion::Future get_future();
  void set_region_field(std::shared_ptr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future);

 public:
  std::shared_ptr<LogicalStore> promote(int32_t extra_dim, size_t dim_size) const;
  std::shared_ptr<LogicalStore> project(int32_t dim, int64_t index) const;

 public:
  std::shared_ptr<LogicalStorePartition> partition_by_tiling(Shape tile_shape);

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  std::unique_ptr<Projection> create_projection(const Partition* partition, int32_t launch_ndim);
  std::shared_ptr<Partition> find_or_create_key_partition();
  void set_key_partition(const Partition* partition);
  void reset_key_partition();

 private:
  std::shared_ptr<LogicalStorePartition> create_partition(std::shared_ptr<Partition> partition);

 private:
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
  std::shared_ptr<Partition> key_partition_;
  std::shared_ptr<Store> mapped_{nullptr};
};

class LogicalStorePartition {
 public:
  LogicalStorePartition(std::shared_ptr<StoragePartition> storage_partition,
                        std::shared_ptr<LogicalStore> store);

 public:
  std::shared_ptr<StoragePartition> storage_partition() const { return storage_partition_; }
  std::shared_ptr<LogicalStore> store() const { return store_; }

 private:
  std::shared_ptr<StoragePartition> storage_partition_;
  std::shared_ptr<LogicalStore> store_;
};

}  // namespace detail
}  // namespace legate
