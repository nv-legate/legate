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

#include "core/mapping/operation.h"
#include "core/mapping/detail/operation.h"

namespace legate::mapping {

int64_t Task::task_id() const { return impl_->task_id(); }

namespace {

template <typename Stores>
std::vector<Store> convert_stores(const Stores& stores)
{
  std::vector<Store> result;
  for (auto& store : stores) { result.emplace_back(&store); }
  return std::move(result);
}

}  // namespace

std::vector<Store> Task::inputs() const { return convert_stores(impl_->inputs()); }

std::vector<Store> Task::outputs() const { return convert_stores(impl_->outputs()); }

std::vector<Store> Task::reductions() const { return convert_stores(impl_->reductions()); }

const std::vector<Scalar>& Task::scalars() const { return impl_->scalars(); }

Store Task::input(uint32_t index) const { return Store(&impl_->inputs().at(index)); }

Store Task::output(uint32_t index) const { return Store(&impl_->outputs().at(index)); }

Store Task::reduction(uint32_t index) const { return Store(&impl_->reductions().at(index)); }

size_t Task::num_inputs() const { return impl_->inputs().size(); }

size_t Task::num_outputs() const { return impl_->outputs().size(); }

size_t Task::num_reductions() const { return impl_->reductions().size(); }

Task::Task(detail::Task* impl) : impl_(impl) {}

// The impl is owned by the caller, so we don't need to deallocate it
Task::~Task() {}

}  // namespace legate::mapping
