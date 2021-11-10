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
#include <unordered_map>

#include "core/data/logical_store.h"
#include "legion.h"

namespace legate {

class Constraint;
class ConstraintGraph;
class LibraryContext;
class Runtime;
class Scalar;
class Strategy;
class Variable;

class Operation {
 public:
  Operation(Runtime* runtime, LibraryContext* library, uint64_t unique_id, int64_t mapper_id);

 public:
  void add_input(LogicalStore store, std::shared_ptr<Variable> partition);
  void add_output(LogicalStore store, std::shared_ptr<Variable> partition);
  void add_reduction(LogicalStore store,
                     Legion::ReductionOpID redop,
                     std::shared_ptr<Variable> partition);

 public:
  std::shared_ptr<Variable> declare_partition(LogicalStore store);
  LogicalStore find_store(std::shared_ptr<Variable> variable) const;
  void add_constraint(std::shared_ptr<Constraint> constraint);
  std::shared_ptr<ConstraintGraph> constraints() const;

 public:
  virtual void launch(Strategy* strategy) = 0;

 public:
  virtual std::string to_string() const = 0;

 protected:
  Runtime* runtime_;
  LibraryContext* library_;
  uint64_t unique_id_;
  int64_t mapper_id_;

 protected:
  using Store = std::pair<LogicalStore, std::shared_ptr<Variable>>;

 protected:
  std::vector<Store> inputs_{};
  std::vector<Store> outputs_{};
  std::vector<Store> reductions_{};
  std::vector<Legion::ReductionOpID> reduction_ops_{};

 private:
  uint32_t next_part_id_{0};

 private:
  std::unordered_map<std::shared_ptr<Variable>, LogicalStore> store_mappings_;
  std::shared_ptr<ConstraintGraph> constraints_;
};

class Task : public Operation {
 public:
  Task(Runtime* runtime,
       LibraryContext* library,
       int64_t task_id,
       uint64_t unique_id,
       int64_t mapper_id = 0);

 public:
  void add_scalar_arg(const Scalar& scalar);

 public:
  virtual void launch(Strategy* strategy) override;

 public:
  virtual std::string to_string() const override;

 private:
  int64_t task_id_;
  std::vector<Scalar> scalars_{};
};

}  // namespace legate
