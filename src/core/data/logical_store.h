/* Copyright 2021 NVIDIA Corporation
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

#include <memory>

#include "legion.h"

#include "core/data/transform.h"
#include "core/utilities/typedefs.h"

namespace legate {

class Runtime;
class Store;
class LibraryContext;
class Partition;
class Projection;

class LogicalRegionField {
 public:
  LogicalRegionField() {}
  LogicalRegionField(Runtime* runtime, const Legion::LogicalRegion& lr, Legion::FieldID fid);

 public:
  LogicalRegionField(const LogicalRegionField& other) = default;
  LogicalRegionField& operator=(const LogicalRegionField& other) = default;

 public:
  int32_t dim() const;
  Legion::LogicalRegion region() const { return lr_; }
  Legion::FieldID field_id() const { return fid_; }

  // public:
  // RegionField map();

 public:
  // template <int32_t DIM>
  // Legion::Rect<DIM> shape() const;
  Legion::Domain domain() const;

 private:
  Runtime* runtime_{nullptr};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{-1U};
};

class LogicalStore {
 public:
  LogicalStore() {}
  LogicalStore(Runtime* runtime,
               LegateTypeCode code,
               std::vector<int64_t> extents,
               std::shared_ptr<StoreTransform> transform = nullptr);

 public:
  LogicalStore(const LogicalStore& other) = default;
  LogicalStore& operator=(const LogicalStore& other) = default;

 public:
  LogicalStore(LogicalStore&& other) = default;
  LogicalStore& operator=(LogicalStore&& other) = default;

 public:
  int32_t dim() const;
  LegateTypeCode code() const { return code_; }
  Legion::Domain domain() const;

 public:
  bool has_storage() const { return nullptr != region_field_; }
  std::shared_ptr<LogicalRegionField> get_storage();
  std::shared_ptr<LogicalRegionField> get_storage_unsafe() const;

 private:
  void create_storage();

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  std::unique_ptr<Projection> find_or_create_partition(const Partition* partition);

 private:
  Runtime* runtime_{nullptr};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  std::vector<int64_t> extents_;
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  std::shared_ptr<StoreTransform> transform_{nullptr};
  std::shared_ptr<Store> mapped_{nullptr};
};

}  // namespace legate
