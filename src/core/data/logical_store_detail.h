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

class LogicalStore {
 public:
  LogicalStore(LegateTypeCode code, tuple<size_t> extents);
  LogicalStore(LegateTypeCode code,
               tuple<size_t> extents,
               std::shared_ptr<LogicalStore> parent,
               std::shared_ptr<TransformStack> transform);
  LogicalStore(LegateTypeCode code, const void* data);

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
  bool scalar() const;
  int32_t dim() const;
  LegateTypeCode code() const;
  Legion::Domain domain() const;
  const std::vector<size_t>& extents() const;
  size_t volume() const;

 public:
  bool has_storage() const;
  std::shared_ptr<LogicalRegionField> get_storage();
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
  std::unique_ptr<Partition> find_or_create_key_partition();

 private:
  std::unique_ptr<Partition> invert_partition(const Partition* partition) const;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const;
  Legion::ProjectionID compute_projection() const;

 public:
  void pack(BufferBuilder& buffer) const;

 private:
  void pack_transform(BufferBuilder& buffer) const;

 private:
  bool scalar_{false};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  tuple<size_t> extents_;
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  Legion::Future future_{};
  std::shared_ptr<LogicalStore> parent_{nullptr};
  std::shared_ptr<TransformStack> transform_{nullptr};
  std::shared_ptr<Store> mapped_{nullptr};
};

}  // namespace detail
}  // namespace legate
