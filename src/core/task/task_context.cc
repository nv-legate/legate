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

#include "core/task/task_context.h"
#include "core/task/detail/task_context.h"

namespace legate {

std::vector<Store>& TaskContext::inputs() { return impl_->inputs(); }

std::vector<Store>& TaskContext::outputs() { return impl_->outputs(); }

std::vector<Store>& TaskContext::reductions() { return impl_->reductions(); }

std::vector<Scalar>& TaskContext::scalars() { return impl_->scalars(); }

std::vector<comm::Communicator>& TaskContext::communicators() { return impl_->communicators(); }

bool TaskContext::is_single_task() const { return impl_->is_single_task(); }

bool TaskContext::can_raise_exception() const { return impl_->can_raise_exception(); }

DomainPoint TaskContext::get_task_index() const { return impl_->get_task_index(); }

Domain TaskContext::get_launch_domain() const { return impl_->get_launch_domain(); }

mapping::Machine TaskContext::machine() const { return mapping::Machine(impl_->machine()); }

const std::string& TaskContext::get_provenance() const { return impl_->get_provenance(); }

TaskContext::TaskContext(detail::TaskContext* impl) : impl_(impl) {}

// No need to delete impl_, as it's owned by the caller
TaskContext::~TaskContext() {}

TaskContext::TaskContext(const TaskContext&) = default;

TaskContext& TaskContext::operator=(const TaskContext&) = default;

}  // namespace legate
