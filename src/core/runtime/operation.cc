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

#include <sstream>
#include <unordered_set>

#include "core/data/scalar.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_graph.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/launcher.h"
#include "core/runtime/operation.h"
#include "core/runtime/req_analyzer.h"
#include "core/runtime/runtime.h"

namespace legate {

Operation::Operation(LibraryContext* library, uint64_t unique_id, int64_t mapper_id)
  : library_(library),
    unique_id_(unique_id),
    mapper_id_(mapper_id),
    constraints_(std::make_shared<ConstraintGraph>())
{
}

Operation::~Operation() {}

void Operation::add_input(LogicalStore store, std::shared_ptr<Variable> partition)
{
  auto p_store = store.impl();
  constraints_->add_variable(partition);
  inputs_.push_back(Store(p_store.get(), std::move(partition)));
  all_stores_.insert(std::move(p_store));
}

void Operation::add_output(LogicalStore store, std::shared_ptr<Variable> partition)
{
  auto p_store = store.impl();
  constraints_->add_variable(partition);
  outputs_.push_back(Store(p_store.get(), std::move(partition)));
  all_stores_.insert(std::move(p_store));
}

void Operation::add_reduction(LogicalStore store,
                              Legion::ReductionOpID redop,
                              std::shared_ptr<Variable> partition)
{
  auto p_store = store.impl();
  constraints_->add_variable(partition);
  reductions_.push_back(Store(p_store.get(), std::move(partition)));
  reduction_ops_.push_back(redop);
  all_stores_.insert(std::move(p_store));
}

std::shared_ptr<Variable> Operation::declare_partition(LogicalStore store)
{
  // TODO: Variable doesn't need a store to be created, so there's some redundancy in this function.
  // Will clean it up once the refactoring for logical store is done
  auto p_store  = store.impl();
  auto variable = std::make_shared<Variable>(this, next_part_id_++);
  store_mappings_.emplace(std::make_pair(variable, p_store.get()));
  all_stores_.insert(std::move(p_store));
  return std::move(variable);
}

detail::LogicalStore* Operation::find_store(std::shared_ptr<Variable> variable) const
{
  auto finder = store_mappings_.find(variable);
  assert(store_mappings_.end() != finder);
  return finder->second;
}

void Operation::add_constraint(std::shared_ptr<Constraint> constraint)
{
  constraints_->add_constraint(constraint);
}

std::shared_ptr<ConstraintGraph> Operation::constraints() const { return constraints_; }

Task::Task(LibraryContext* library, int64_t task_id, uint64_t unique_id, int64_t mapper_id /*=0*/)
  : Operation(library, unique_id, mapper_id), task_id_(task_id)
{
}

Task::~Task() {}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

void Task::launch(Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  TaskLauncher launcher(library_, task_id_, mapper_id_);

  for (auto& pair : inputs_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store);
    launcher.add_input(store, std::move(proj));
  }
  for (auto& pair : outputs_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store);
    launcher.add_output(store, std::move(proj));
  }
  uint32_t idx = 0;
  for (auto& pair : reductions_) {
    auto& store = pair.first;
    auto& var   = pair.second;
    auto proj   = strategy[var]->get_projection(store);
    auto redop  = reduction_ops_[idx++];
    proj->set_reduction_op(redop);
    launcher.add_reduction(store, std::move(proj));
  }
  for (auto& scalar : scalars_) launcher.add_scalar(scalar);

  if (strategy.parallel(this))
    launcher.execute(strategy.launch_domain(this));
  else
    launcher.execute_single();
}

std::string Task::to_string() const
{
  std::stringstream ss;
  ss << library_->get_task_name(task_id_) << ":" << unique_id_;
  return ss.str();
}

}  // namespace legate
