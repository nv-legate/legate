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
#include <valarray>

#include "legion.h"

#include "core/data/shape.h"
#include "core/data/transform.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

namespace legate {

class BufferBuilder;
class LibraryContext;
class LogicalStorePartition;
class Partition;
class Projection;
class Runtime;
class Store;

namespace detail {

class LogicalStore;
class LogicalStorePartition;

}  // namespace detail

class LogicalStore {
 private:
  friend class Runtime;
  friend class LogicalStorePartition;
  LogicalStore(std::shared_ptr<detail::LogicalStore>&& impl);

 public:
  LogicalStore(const LogicalStore& other)            = default;
  LogicalStore& operator=(const LogicalStore& other) = default;

 public:
  LogicalStore(LogicalStore&& other)            = default;
  LogicalStore& operator=(LogicalStore&& other) = default;

 public:
  int32_t dim() const;
  const Type& type() const;
  const Shape& extents() const;
  size_t volume() const;
  bool unbound() const;
  bool transformed() const;

 public:
  LogicalStore promote(int32_t extra_dim, size_t dim_size) const;
  LogicalStore project(int32_t dim, int64_t index) const;
  LogicalStore slice(int32_t dim, std::slice sl) const;
  LogicalStore transpose(std::vector<int32_t>&& axes) const;
  LogicalStore delinearize(int32_t dim, std::vector<int64_t>&& sizes) const;

 public:
  LogicalStorePartition partition_by_tiling(std::vector<size_t> tile_shape) const;

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  void set_key_partition(const Partition* partition);

 public:
  std::shared_ptr<detail::LogicalStore> impl() const { return impl_; }

 private:
  std::shared_ptr<detail::LogicalStore> impl_{nullptr};
};

class LogicalStorePartition {
 private:
  friend class LogicalStore;
  LogicalStorePartition(std::shared_ptr<detail::LogicalStorePartition>&& impl);

 public:
  LogicalStore store() const;
  std::shared_ptr<Partition> partition() const;

 private:
  std::shared_ptr<detail::LogicalStorePartition> impl_{nullptr};
};

}  // namespace legate
