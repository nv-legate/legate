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

#include "core/operation/detail/operation.h"

#include "core/partitioning/constraint.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

Operation::Operation(uint64_t unique_id, mapping::MachineDesc&& machine)
  : unique_id_(unique_id),
    machine_(std::move(machine)),
    provenance_(Runtime::get_runtime()->provenance_manager()->get_provenance())
{
}

const Variable* Operation::find_or_declare_partition(std::shared_ptr<LogicalStore> store)
{
  auto finder = part_mappings_.find(store);
  if (finder != part_mappings_.end()) return finder->second;
  const auto* symb = declare_partition();
  part_mappings_.insert({std::move(store), symb});
  return symb;
}

const Variable* Operation::declare_partition()
{
  partition_symbols_.emplace_back(new Variable(this, next_part_id_++));
  return partition_symbols_.back().get();
}

LogicalStore* Operation::find_store(const Variable* part_symb) const
{
  return store_mappings_.at(*part_symb);
}

void Operation::record_partition(const Variable* variable, std::shared_ptr<LogicalStore> store)
{
  auto finder = store_mappings_.find(*variable);
  if (finder != store_mappings_.end()) {
    if (finder->second->id() != store->id())
      throw std::invalid_argument("Variable " + variable->to_string() +
                                  " is already assigned to another store");
    return;
  }
  if (part_mappings_.find(store) == part_mappings_.end()) part_mappings_.insert({store, variable});
  store_mappings_[*variable] = store.get();
  all_stores_.insert(std::move(store));
}

}  // namespace legate::detail
