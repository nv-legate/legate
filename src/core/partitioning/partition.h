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

namespace legate {

class LogicalStore;
class Projection;
class Runtime;

struct Partition {
 public:
  virtual bool is_complete_for(const LogicalStore* store) const = 0;
  virtual bool is_disjoint_for(const LogicalStore* store) const = 0;

 public:
  virtual Legion::LogicalPartition construct(const LogicalStore* store,
                                             bool disjoint = false,
                                             bool complete = false) const       = 0;
  virtual std::unique_ptr<Projection> get_projection(LogicalStore* store) const = 0;

 public:
  virtual bool has_launch_domain() const       = 0;
  virtual Legion::Domain launch_domain() const = 0;
};

struct PartitioningFunctor {
 public:
  virtual Legion::IndexPartition construct(Legion::Runtime* legion_runtime,
                                           Legion::Context legion_context,
                                           const Legion::IndexSpace& parent,
                                           const Legion::IndexSpace& color_space,
                                           Legion::PartitionKind kind) const = 0;
};

std::unique_ptr<Partition> create_no_partition(Runtime* runtime);

std::unique_ptr<Partition> create_tiling(Runtime* runtime,
                                         std::vector<size_t>&& tile_shape,
                                         std::vector<size_t>&& color_shape,
                                         std::vector<size_t>&& offsets = {});

}  // namespace legate
