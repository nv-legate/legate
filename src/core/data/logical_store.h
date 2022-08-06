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

class BufferBuilder;
class LibraryContext;
class LogicalRegionField;
class Partition;
class Projection;
class Runtime;
class Store;

namespace detail {

class LogicalStore;

}  // namespace detail

class LogicalStore {
 private:
  friend class Runtime;
  LogicalStore(std::shared_ptr<detail::LogicalStore>&& impl);

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
  LogicalStore promote(int32_t extra_dim, size_t dim_size) const;

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  std::shared_ptr<detail::LogicalStore> impl() const { return impl_; }

 private:
  std::shared_ptr<detail::LogicalStore> impl_{nullptr};
};

}  // namespace legate
