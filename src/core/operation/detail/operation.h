/* Copyright 2023 NVIDIA Corporation
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

#include "core/data/detail/logical_store.h"
#include "core/mapping/detail/machine.h"
#include "core/operation/detail/projection.h"

namespace legate::detail {
class ConstraintSolver;
class LogicalStore;
class Strategy;
class Variable;

class Operation {
 protected:
  struct StoreArg {
    LogicalStore* store;
    const Variable* variable;
  };
  Operation(uint64_t unique_id, mapping::detail::Machine&& machine);

 public:
  virtual ~Operation() {}

 public:
  virtual void validate()                              = 0;
  virtual void add_to_solver(ConstraintSolver& solver) = 0;
  virtual void launch(Strategy* strategy)              = 0;
  virtual std::string to_string() const                = 0;

 public:
  const Variable* find_or_declare_partition(std::shared_ptr<LogicalStore> store);
  const Variable* declare_partition();
  std::shared_ptr<LogicalStore> find_store(const Variable* variable) const;

 public:
  const mapping::detail::Machine& machine() const { return machine_; }
  const std::string& provenance() const { return provenance_; }

 protected:
  void record_partition(const Variable* variable, std::shared_ptr<LogicalStore> store);
  // Helper methods
  std::unique_ptr<ProjectionInfo> create_projection_info(const Strategy& strategy,
                                                         const Domain* launch_domain,
                                                         const StoreArg& arg) const;

 protected:
  uint64_t unique_id_;

 protected:
  std::set<std::shared_ptr<LogicalStore>> all_stores_{};
  std::vector<StoreArg> inputs_{};
  std::vector<StoreArg> outputs_{};
  std::vector<StoreArg> reductions_{};
  std::vector<Legion::ReductionOpID> reduction_ops_{};

 protected:
  uint32_t next_part_id_{0};
  std::vector<std::unique_ptr<Variable>> partition_symbols_{};
  std::map<const Variable, std::shared_ptr<LogicalStore>> store_mappings_{};
  std::map<std::shared_ptr<LogicalStore>, const Variable*> part_mappings_{};
  std::string provenance_;
  mapping::detail::Machine machine_;
};

}  // namespace legate::detail
