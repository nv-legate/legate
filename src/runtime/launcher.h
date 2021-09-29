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

class BufferBuilder;
class RegionReq;
class RequirementAnalyzer;
class ArgWrapper;

class LogicalStore;
class Scalar;

class Runtime;
class LibraryContext;

class Projection {
 protected:
  using SingleTask = Legion::TaskLauncher*;

 protected:
  Projection() {}
  Projection(Legion::ReductionOpID redop);

 public:
  virtual ~Projection() {}

 public:
  virtual void add(SingleTask task,
                   const RegionReq& req,
                   const std::vector<Legion::FieldID>& fields) const = 0;

 public:
  std::unique_ptr<Legion::ReductionOpID> redop{nullptr};
};

class Broadcast : public Projection {
 public:
  Broadcast();
  Broadcast(Legion::ReductionOpID redop);

 public:
  virtual ~Broadcast() {}

 public:
  virtual void add(SingleTask task,
                   const RegionReq& req,
                   const std::vector<Legion::FieldID>& fields) const override;
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
  using LogicalStoreP = std::shared_ptr<LogicalStore>;
  using ProjectionP   = std::unique_ptr<Projection>;
  using SingleTask    = Legion::TaskLauncher*;

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
  void add_input(LogicalStoreP store, ProjectionP proj, uint64_t tag = 0);
  void add_output(LogicalStoreP store, ProjectionP proj, uint64_t tag = 0);
  void add_reduction(LogicalStoreP store,
                     ProjectionP proj,
                     uint64_t tag    = 0,
                     bool read_write = false);

 private:
  void add_store(std::vector<ArgWrapper*>& args,
                 LogicalStoreP store,
                 ProjectionP proj,
                 Legion::PrivilegeMode privilege,
                 uint64_t tag);

 public:
  void execute_single();

 private:
  void pack_args(const std::vector<ArgWrapper*>& args);
  SingleTask build_single_task();

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
