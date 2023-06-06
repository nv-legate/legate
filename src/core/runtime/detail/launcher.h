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

#include "core/mapping/machine.h"

namespace legate {
class BufferBuilder;
class LibraryContext;
class Scalar;
}  // namespace legate

namespace legate::detail {

class ArgWrapper;
class LogicalStore;
class Projection;
class OutputRegionArg;
class OutputRequirementAnalyzer;
class RequirementAnalyzer;

class TaskLauncher {
 public:
  TaskLauncher(const LibraryContext* library,
               const mapping::MachineDesc& machine,
               int64_t task_id,
               int64_t tag = 0);
  TaskLauncher(const LibraryContext* library,
               const mapping::MachineDesc& machine,
               const std::string& provenance,
               int64_t task_id,
               int64_t tag = 0);
  ~TaskLauncher();

 private:
  void initialize();

 public:
  int64_t legion_task_id() const;
  int64_t legion_mapper_id() const;

 public:
  void add_scalar(const Scalar& scalar);
  void add_input(LogicalStore* store,
                 std::unique_ptr<Projection> proj,
                 Legion::MappingTagID tag  = 0,
                 Legion::RegionFlags flags = LEGION_NO_FLAG);
  void add_output(LogicalStore* store,
                  std::unique_ptr<Projection> proj,
                  Legion::MappingTagID tag  = 0,
                  Legion::RegionFlags flags = LEGION_NO_FLAG);
  void add_reduction(LogicalStore* store,
                     std::unique_ptr<Projection> proj,
                     bool read_write           = false,
                     Legion::MappingTagID tag  = 0,
                     Legion::RegionFlags flags = LEGION_NO_FLAG);
  void add_unbound_output(LogicalStore* store,
                          Legion::FieldSpace field_space,
                          Legion::FieldID field_id);

 public:
  void add_future(const Legion::Future& future);
  void add_future_map(const Legion::FutureMap& future_map);
  void add_communicator(const Legion::FutureMap& communicator);

 public:
  void set_side_effect(bool has_side_effect) { has_side_effect_ = has_side_effect; }
  void set_concurrent(bool is_concurrent) { concurrent_ = is_concurrent; }
  void set_insert_barrier(bool insert_barrier) { insert_barrier_ = insert_barrier; }
  void throws_exception(bool can_throw_exception) { can_throw_exception_ = can_throw_exception; }

 private:
  void add_store(std::vector<ArgWrapper*>& args,
                 LogicalStore* store,
                 std::unique_ptr<Projection> proj,
                 Legion::PrivilegeMode privilege,
                 Legion::MappingTagID tag,
                 Legion::RegionFlags flags);

 public:
  Legion::FutureMap execute(const Legion::Domain& launch_domain);
  Legion::Future execute_single();

 private:
  void pack_args(const std::vector<ArgWrapper*>& args);
  void pack_sharding_functor_id();
  std::unique_ptr<Legion::IndexTaskLauncher> build_index_task(const Legion::Domain& launch_domain);
  std::unique_ptr<Legion::TaskLauncher> build_single_task();
  void bind_region_fields_to_unbound_stores();
  void post_process_unbound_stores();
  void post_process_unbound_stores(const Legion::FutureMap& result,
                                   const Legion::Domain& launch_domain);

 private:
  const LibraryContext* library_;
  int64_t task_id_;
  int64_t tag_;
  mapping::MachineDesc machine_;
  std::string provenance_;

 private:
  bool has_side_effect_{true};
  bool concurrent_{false};
  bool insert_barrier_{false};
  bool can_throw_exception_{false};

 private:
  std::vector<ArgWrapper*> inputs_;
  std::vector<ArgWrapper*> outputs_;
  std::vector<ArgWrapper*> reductions_;
  std::vector<ArgWrapper*> scalars_;
  std::vector<Legion::Future> futures_;
  std::vector<OutputRegionArg*> unbound_stores_;
  std::vector<Legion::FutureMap> future_maps_;
  std::vector<Legion::FutureMap> communicators_;

 private:
  RequirementAnalyzer* req_analyzer_;
  OutputRequirementAnalyzer* out_analyzer_;
  BufferBuilder* buffer_;
  BufferBuilder* mapper_arg_;
  std::vector<Legion::OutputRequirement> output_requirements_;
};

}  // namespace legate::detail
