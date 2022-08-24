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

class ArgWrapper;
class BufferBuilder;
class LibraryContext;
class LogicalStore;
class Projection;
class OutputRegionArg;
class OutputRequirementAnalyzer;
class RequirementAnalyzer;
class Scalar;

namespace detail {

class LogicalStore;

}  // namespace detail

class TaskLauncher {
 public:
  TaskLauncher(LibraryContext* library, int64_t task_id, int64_t mapper_id = 0, int64_t tag = 0);
  ~TaskLauncher();

 public:
  int64_t legion_task_id() const;
  int64_t legion_mapper_id() const;

 public:
  void add_scalar(const Scalar& scalar);
  void add_input(detail::LogicalStore* store,
                 std::unique_ptr<Projection> proj,
                 Legion::MappingTagID tag  = 0,
                 Legion::RegionFlags flags = LEGION_NO_FLAG);
  void add_output(detail::LogicalStore* store,
                  std::unique_ptr<Projection> proj,
                  Legion::MappingTagID tag  = 0,
                  Legion::RegionFlags flags = LEGION_NO_FLAG);
  void add_reduction(detail::LogicalStore* store,
                     std::unique_ptr<Projection> proj,
                     Legion::MappingTagID tag  = 0,
                     Legion::RegionFlags flags = LEGION_NO_FLAG,
                     bool read_write           = false);
  void add_unbound_output(detail::LogicalStore* store,
                          Legion::FieldSpace field_space,
                          Legion::FieldID field_id);

 private:
  void add_store(std::vector<ArgWrapper*>& args,
                 detail::LogicalStore* store,
                 std::unique_ptr<Projection> proj,
                 Legion::PrivilegeMode privilege,
                 Legion::MappingTagID tag,
                 Legion::RegionFlags flags);

 public:
  void execute(const Legion::Domain& launch_domain);
  void execute_single();

 private:
  void pack_args(const std::vector<ArgWrapper*>& args);
  Legion::IndexTaskLauncher* build_index_task(const Legion::Domain& launch_domain);
  Legion::TaskLauncher* build_single_task();
  void bind_region_fields_to_unbound_stores();

 private:
  LibraryContext* library_;
  int64_t task_id_;
  int64_t mapper_id_;
  int64_t tag_;

 private:
  std::vector<ArgWrapper*> inputs_;
  std::vector<ArgWrapper*> outputs_;
  std::vector<ArgWrapper*> reductions_;
  std::vector<ArgWrapper*> scalars_;
  std::vector<Legion::Future> futures_;
  std::vector<OutputRegionArg*> unbound_stores_;

 private:
  RequirementAnalyzer* req_analyzer_;
  OutputRequirementAnalyzer* out_analyzer_;
  BufferBuilder* buffer_;
  std::vector<Legion::OutputRequirement> output_requirements_;
};

}  // namespace legate
