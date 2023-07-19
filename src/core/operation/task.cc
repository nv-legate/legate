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

#include "core/operation/task.h"

#include "core/operation/detail/task.h"

namespace legate {

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

void AutoTask::add_input(LogicalStore store, const Variable* partition_symbol)
{
  impl_->add_input(store.impl(), partition_symbol);
}

void AutoTask::add_output(LogicalStore store, const Variable* partition_symbol)
{
  impl_->add_output(store.impl(), partition_symbol);
}

void AutoTask::add_reduction(LogicalStore store,
                             ReductionOpKind redop,
                             const Variable* partition_symbol)
{
  impl_->add_reduction(store.impl(), static_cast<int32_t>(redop), partition_symbol);
}

void AutoTask::add_reduction(LogicalStore store, int32_t redop, const Variable* partition_symbol)
{
  impl_->add_reduction(store.impl(), redop, partition_symbol);
}

void AutoTask::add_scalar_arg(const Scalar& scalar) { impl_->add_scalar_arg(scalar); }

void AutoTask::add_scalar_arg(Scalar&& scalar) { impl_->add_scalar_arg(scalar); }

void AutoTask::add_constraint(std::unique_ptr<Constraint> constraint)
{
  impl_->add_constraint(std::move(constraint));
}

const Variable* AutoTask::find_or_declare_partition(LogicalStore store)
{
  return impl_->find_or_declare_partition(store.impl());
}

const Variable* AutoTask::declare_partition() { return impl_->declare_partition(); }

const mapping::MachineDesc& AutoTask::machine() const { return impl_->machine(); }

const std::string& AutoTask::provenance() const { return impl_->provenance(); }

void AutoTask::set_concurrent(bool concurrent) { impl_->set_concurrent(concurrent); }

void AutoTask::set_side_effect(bool has_side_effect) { impl_->set_side_effect(has_side_effect); }

void AutoTask::throws_exception(bool can_throw_exception)
{
  impl_->throws_exception(can_throw_exception);
}

void AutoTask::add_communicator(const std::string& name) { impl_->add_communicator(name); }

AutoTask::AutoTask(AutoTask&&) = default;

AutoTask& AutoTask::operator=(AutoTask&&) = default;

AutoTask::~AutoTask() {}

AutoTask::AutoTask(std::unique_ptr<detail::AutoTask> impl) : impl_(std::move(impl)) {}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

void ManualTask::add_input(LogicalStore store) { impl_->add_input(store.impl()); }

void ManualTask::add_input(LogicalStorePartition store_partition)
{
  impl_->add_input(store_partition.impl());
}

void ManualTask::add_output(LogicalStore store) { impl_->add_output(store.impl()); }

void ManualTask::add_output(LogicalStorePartition store_partition)
{
  impl_->add_output(store_partition.impl());
}

void ManualTask::add_reduction(LogicalStore store, ReductionOpKind redop)
{
  impl_->add_reduction(store.impl(), static_cast<int32_t>(redop));
}

void ManualTask::add_reduction(LogicalStore store, int32_t redop)
{
  impl_->add_reduction(store.impl(), redop);
}

void ManualTask::add_reduction(LogicalStorePartition store_partition, ReductionOpKind redop)
{
  impl_->add_reduction(store_partition.impl(), static_cast<int32_t>(redop));
}

void ManualTask::add_reduction(LogicalStorePartition store_partition, int32_t redop)
{
  impl_->add_reduction(store_partition.impl(), redop);
}

void ManualTask::add_scalar_arg(const Scalar& scalar) { impl_->add_scalar_arg(scalar); }

void ManualTask::add_scalar_arg(Scalar&& scalar) { impl_->add_scalar_arg(scalar); }

const mapping::MachineDesc& ManualTask::machine() const { return impl_->machine(); }

const std::string& ManualTask::provenance() const { return impl_->provenance(); }

void ManualTask::set_concurrent(bool concurrent) { impl_->set_concurrent(concurrent); }

void ManualTask::set_side_effect(bool has_side_effect) { impl_->set_side_effect(has_side_effect); }

void ManualTask::throws_exception(bool can_throw_exception)
{
  impl_->throws_exception(can_throw_exception);
}

void ManualTask::add_communicator(const std::string& name) { impl_->add_communicator(name); }

ManualTask::ManualTask(ManualTask&&) = default;

ManualTask& ManualTask::operator=(ManualTask&&) = default;

ManualTask::~ManualTask() {}

ManualTask::ManualTask(std::unique_ptr<detail::ManualTask> impl) : impl_(std::move(impl)) {}

}  // namespace legate
