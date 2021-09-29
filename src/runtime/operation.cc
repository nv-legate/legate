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

#include "runtime/operation.h"
#include "data/logical_store.h"
#include "data/scalar.h"
#include "runtime/context.h"
#include "runtime/launcher.h"
#include "runtime/runtime.h"

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

Task::Task(Runtime* runtime, LibraryContext* library, int64_t task_id, int64_t mapper_id /*=0*/)
  : Operation(runtime, library, mapper_id), task_id_(task_id)
{
}

void Task::add_scalar_arg(const Scalar& scalar) { scalars_.push_back(scalar); }

void Task::launch() const
{
  TaskLauncher launcher(runtime_, library_, task_id_, mapper_id_);

  for (auto& input : inputs_) launcher.add_input(input, std::make_unique<Broadcast>());
  for (auto& output : outputs_) launcher.add_output(output, std::make_unique<Broadcast>());
  for (auto& pair : reductions_)
    launcher.add_reduction(pair.first, std::make_unique<Broadcast>(pair.second));
  for (auto& scalar : scalars_) launcher.add_scalar(scalar);

  launcher.execute_single();
}

}  // namespace legate
