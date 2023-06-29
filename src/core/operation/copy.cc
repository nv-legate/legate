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

#include "core/operation/copy.h"

#include "core/operation/detail/copy.h"

namespace legate {

void Copy::add_input(LogicalStore store) { impl_->add_input(store.impl()); }

void Copy::add_output(LogicalStore store) { impl_->add_output(store.impl()); }

void Copy::add_reduction(LogicalStore store, Legion::ReductionOpID redop)
{
  impl_->add_reduction(store.impl(), redop);
}

void Copy::add_source_indirect(LogicalStore store) { impl_->add_source_indirect(store.impl()); }

void Copy::add_target_indirect(LogicalStore store) { impl_->add_target_indirect(store.impl()); }

void Copy::set_source_indirect_out_of_range(bool flag)
{
  impl_->set_source_indirect_out_of_range(flag);
}

void Copy::set_target_indirect_out_of_range(bool flag)
{
  impl_->set_target_indirect_out_of_range(flag);
}

Copy::~Copy() {}

Copy::Copy(std::unique_ptr<detail::Copy> impl) : impl_(std::move(impl)) {}

}  // namespace legate
