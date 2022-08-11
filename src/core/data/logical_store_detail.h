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

class Storage {
 public:
  enum class Kind : int32_t {
    REGION_FIELD = 0,
    FUTURE       = 1,
  };

 public:
  // Create a RegionField-backed storage. Initialized lazily.
  Storage(tuple<size_t> extents, LegateTypeCode code);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(tuple<size_t> extents, LegateTypeCode code, const Legion::Future& future);

 public:
  const tuple<size_t>& extents() const { return extents_; }
  size_t volume() const { return volume_; }
  int32_t dim();
  LegateTypeCode code() const { return code_; }
  Kind kind() const { return kind_; }

 public:
  LogicalRegionField* get_region_field();
  Legion::Future get_future() const;

 public:
  RegionField map(LibraryContext* context);

 public:
  Partition* find_or_create_key_partition();
  void set_key_partition(std::unique_ptr<Partition>&& key_partition);
  void reset_key_partition();
  Legion::LogicalPartition find_or_create_legion_partition(const Partition* partition);

 private:
  tuple<size_t> extents_;
  size_t volume_;
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  Kind kind_{Kind::REGION_FIELD};
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  Legion::Future future_{};

 private:
  std::unique_ptr<Partition> key_partition_{nullptr};
};

class LogicalStore {
 public:
  LogicalStore(std::shared_ptr<Storage>&& storage);
  LogicalStore(tuple<size_t>&& extents,
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
  const tuple<size_t>& extents() const;
  size_t volume() const;
  // Size of the backing storage
  size_t storage_size() const;
  int32_t dim() const;
  bool scalar() const;
  LegateTypeCode code() const;

 public:
  LogicalRegionField* get_region_field();
  Legion::Future get_future();

 private:
  void create_storage();

 public:
  std::shared_ptr<LogicalStore> promote(int32_t extra_dim,
                                        size_t dim_size,
                                        std::shared_ptr<LogicalStore> parent) const;

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  std::unique_ptr<Projection> create_projection(const Partition* partition);
  std::shared_ptr<Partition> find_or_create_key_partition();
  void set_key_partition(const Partition* partition);
  void reset_key_partition();

 private:
  Legion::ProjectionID compute_projection() const;

 public:
  void pack(BufferBuilder& buffer) const;

 public:
  std::string to_string() const;

 private:
  uint64_t store_id_;
  tuple<size_t> extents_;
  std::shared_ptr<Storage> storage_;
  std::shared_ptr<TransformStack> transform_;

 private:
  std::shared_ptr<Partition> key_partition_;
  std::shared_ptr<Store> mapped_{nullptr};
};

}  // namespace detail
}  // namespace legate
