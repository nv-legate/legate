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

#include <unordered_set>

#include "core/data/logical_store.h"
#include "core/data/scalar.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/launcher.h"
#include "core/runtime/operation.h"
#include "core/runtime/runtime.h"

namespace legate {

Operation::Operation(Runtime* runtime, LibraryContext* library, int64_t mapper_id)
  : runtime_(runtime), library_(library), mapper_id_(mapper_id)
{
}

void Operation::add_input(LogicalStoreP store) { inputs_.push_back(store); }

void Operation::add_output(LogicalStoreP store) { outputs_.push_back(store); }

void Operation::add_reduction(LogicalStoreP store, Legion::ReductionOpID redop)
{
  reductions_.push_back(Reduction(store, redop));
}

std::vector<LogicalStore*> Operation::all_stores()
{
  std::vector<LogicalStore*> result;
  std::unordered_set<LogicalStore*> added;

  auto add_all = [&](auto& stores) {
    for (auto& store : stores) {
      auto p_store = store.get();
      if (added.find(p_store) == added.end()) {
        result.push_back(p_store);
        added.insert(p_store);
      }
    }
  };

  add_all(inputs_);
  add_all(outputs_);
  for (auto& reduction : reductions_) {
    auto& store  = reduction.first;
    auto p_store = store.get();
    if (added.find(p_store) == added.end()) {
      result.push_back(p_store);
      added.insert(p_store);
    }
  }

  return std::move(result);
}

Task::Task(Runtime* runtime, LibraryContext* library, int64_t task_id, int64_t mapper_id /*=0*/)
  : Operation(runtime, library, mapper_id), task_id_(task_id)
{
}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

void Task::launch(Strategy* strategy) const
{
  TaskLauncher launcher(runtime_, library_, task_id_, mapper_id_);

  for (auto& input : inputs_) launcher.add_input(input, strategy->get_projection(input.get()));
  for (auto& output : outputs_) launcher.add_output(output, strategy->get_projection(output.get()));
  for (auto& pair : reductions_) {
    auto projection = strategy->get_projection(pair.first.get());
    projection->set_reduction_op(pair.second);
    launcher.add_reduction(pair.first, std::move(projection));
  }
  for (auto& scalar : scalars_) launcher.add_scalar(scalar);

  if (strategy->parallel(this))
    launcher.execute(strategy->launch_domain(this));
  else
    launcher.execute_single();
}

}  // namespace legate
