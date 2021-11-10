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
class RegionReq;
class RequirementAnalyzer;
class Runtime;
class Scalar;

class Projection {
 protected:
  Projection() {}
  Projection(Legion::ReductionOpID redop);

 public:
  virtual ~Projection() {}

 public:
  virtual void populate_launcher(Legion::TaskLauncher* task,
                                 const RegionReq& req,
                                 const std::vector<Legion::FieldID>& fields) const = 0;
  virtual void populate_launcher(Legion::IndexTaskLauncher* task,
                                 const RegionReq& req,
                                 const std::vector<Legion::FieldID>& fields) const = 0;

 public:
  void set_reduction_op(Legion::ReductionOpID redop);

 public:
  std::unique_ptr<Legion::ReductionOpID> redop{nullptr};
};

class Replicate : public Projection {
 public:
  Replicate();
  Replicate(Legion::ReductionOpID redop);

 public:
  virtual ~Replicate() {}

 public:
  virtual void populate_launcher(Legion::TaskLauncher* task,
                                 const RegionReq& req,
                                 const std::vector<Legion::FieldID>& fields) const override;
  virtual void populate_launcher(Legion::IndexTaskLauncher* task,
                                 const RegionReq& req,
                                 const std::vector<Legion::FieldID>& fields) const override;
};

class MapPartition : public Projection {
 public:
  MapPartition(Legion::LogicalPartition partition, Legion::ProjectionID proj_id);
  MapPartition(Legion::LogicalPartition partition,
               Legion::ProjectionID proj_id,
               Legion::ReductionOpID redop);

 public:
  virtual ~MapPartition() {}

 public:
  virtual void populate_launcher(Legion::TaskLauncher* task,
                                 const RegionReq& req,
                                 const std::vector<Legion::FieldID>& fields) const override;
  virtual void populate_launcher(Legion::IndexTaskLauncher* task,
                                 const RegionReq& req,
                                 const std::vector<Legion::FieldID>& fields) const override;

 private:
  Legion::LogicalPartition partition_;
  Legion::ProjectionID proj_id_;
};

class RegionReq {
 private:
  using ProjectionP = std::unique_ptr<Projection>;

 public:
  RegionReq(Legion::LogicalRegion region,
            Legion::PrivilegeMode priv,
            ProjectionP proj,
            int64_t tag);

 public:
  Legion::LogicalRegion region;
  Legion::PrivilegeMode priv;
  ProjectionP proj;
  int64_t tag;
};

class TaskLauncher {
 private:
  using ProjectionP = std::unique_ptr<Projection>;

 public:
  TaskLauncher(Runtime* runtime,
               LibraryContext* library,
               int64_t task_id,
               int64_t mapper_id = 0,
               int64_t tag       = 0);
  ~TaskLauncher();

 public:
  int64_t legion_task_id() const;
  int64_t legion_mapper_id() const;

 public:
  void add_scalar(const Scalar& scalar);
  void add_input(LogicalStore store, ProjectionP proj, uint64_t tag = 0);
  void add_output(LogicalStore store, ProjectionP proj, uint64_t tag = 0);
  void add_reduction(LogicalStore store,
                     ProjectionP proj,
                     uint64_t tag    = 0,
                     bool read_write = false);

 private:
  void add_store(std::vector<ArgWrapper*>& args,
                 LogicalStore store,
                 ProjectionP proj,
                 Legion::PrivilegeMode privilege,
                 uint64_t tag);

 public:
  void execute(const Legion::Domain& launch_domain);
  void execute_single();

 private:
  void pack_args(const std::vector<ArgWrapper*>& args);
  Legion::IndexTaskLauncher* build_index_task(const Legion::Domain& launch_domain);
  Legion::TaskLauncher* build_single_task();

 private:
  Runtime* runtime_;
  LibraryContext* library_;
  int64_t task_id_;
  int64_t mapper_id_;
  int64_t tag_;

 private:
  std::vector<ArgWrapper*> inputs_;
  std::vector<ArgWrapper*> outputs_;
  std::vector<ArgWrapper*> reductions_;
  std::vector<ArgWrapper*> scalars_;

 private:
  RequirementAnalyzer* req_analyzer_;
  BufferBuilder* buffer_;
};

}  // namespace legate
