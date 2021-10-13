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

class LibraryContext;
class LogicalStore;
class Runtime;
class Scalar;
class Strategy;

class Operation {
 private:
  using LogicalStoreP = std::shared_ptr<LogicalStore>;
  using Reduction     = std::pair<LogicalStoreP, Legion::ReductionOpID>;

 public:
  Operation(Runtime* runtime, LibraryContext* library, int64_t mapper_id);

 public:
  void add_input(LogicalStoreP store);
  void add_output(LogicalStoreP store);
  void add_reduction(LogicalStoreP store, Legion::ReductionOpID redop);

 public:
  std::vector<LogicalStore*> all_stores();

 public:
  virtual void launch(Strategy* strategy) const = 0;

 protected:
  Runtime* runtime_;
  LibraryContext* library_;
  int64_t mapper_id_;

 protected:
  std::vector<LogicalStoreP> inputs_;
  std::vector<LogicalStoreP> outputs_;
  std::vector<Reduction> reductions_;
};

class Task : public Operation {
 public:
  Task(Runtime* runtime, LibraryContext* library, int64_t task_id, int64_t mapper_id = 0);

 public:
  void add_scalar_arg(const Scalar& scalar);

 public:
  virtual void launch(Strategy* strategy) const override;

 private:
  int64_t task_id_;
  std::vector<Scalar> scalars_{};
};

}  // namespace legate
