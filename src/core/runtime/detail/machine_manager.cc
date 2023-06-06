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

#include "core/runtime/detail/machine_manager.h"

namespace legate::detail {

////////////////////////////////////////////
// legate::MachineManager
////////////////////////////////////////////
const mapping::MachineDesc& MachineManager::get_machine() const
{
#ifdef DEBUG_LEGATE
  assert(machines_.size() > 0);
#endif
  return machines_.back();
}

void MachineManager::push_machine(const mapping::MachineDesc& machine)
{
  machines_.push_back(machine);
}

void MachineManager::push_machine(mapping::MachineDesc&& machine)
{
  machines_.emplace_back(machine);
}

void MachineManager::pop_machine()
{
  if (machines_.size() <= 1) throw std::underflow_error("can't pop from the empty machine stack");
  machines_.pop_back();
}

}  // namespace legate::detail
