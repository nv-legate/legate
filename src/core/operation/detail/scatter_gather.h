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
#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"

namespace legate::detail {

class ConstraintSolver;

class ScatterGather : public Operation {
 public:
  ScatterGather(std::shared_ptr<LogicalStore> target,
                std::shared_ptr<LogicalStore> target_indirect,
                std::shared_ptr<LogicalStore> source,
                std::shared_ptr<LogicalStore> source_indirect,
                int64_t unique_id,
                mapping::MachineDesc&& machine);

 public:
  void set_source_indirect_out_of_range(bool flag);
  void set_target_indirect_out_of_range(bool flag);

 public:
  void validate() override;
  void launch(detail::Strategy* strategy) override;

 public:
  void add_to_solver(detail::ConstraintSolver& solver) override;

 public:
  std::string to_string() const override;

 private:
  bool source_indirect_out_of_range_{true};
  bool target_indirect_out_of_range_{true};
  StoreArg target_;
  StoreArg target_indirect_;
  StoreArg source_;
  StoreArg source_indirect_;
  std::unique_ptr<Constraint> constraint_;
};

}  // namespace legate::detail
