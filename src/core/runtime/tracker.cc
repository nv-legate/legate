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

#include "core/runtime/tracker.h"

#include "core/runtime/machine_manager.h"
#include "core/runtime/provenance_manager.h"
#include "core/runtime/runtime.h"

namespace legate {

//////////////////////////////////////
//  ProvenanceTracker
//////////////////////////////////////

ProvenanceTracker::ProvenanceTracker(const std::string& p)
{
  auto* runtime = Runtime::get_runtime();
  runtime->provenance_manager()->push_provenance(p);
}

ProvenanceTracker::~ProvenanceTracker()
{
  auto* runtime = Runtime::get_runtime();
  runtime->provenance_manager()->pop_provenance();
}

const std::string& ProvenanceTracker::get_current_provenance() const
{
  return Runtime::get_runtime()->provenance_manager()->get_provenance();
}

////////////////////////////////////////////
// legate::MachineTracker
////////////////////////////////////////////

MachineTracker::MachineTracker(const mapping::MachineDesc& machine)
{
  auto* runtime = Runtime::get_runtime();
  auto result   = machine & Runtime::get_runtime()->get_machine();
  if (result.count() == 0)
    throw std::runtime_error("Empty machines cannot be used for resource scoping");
  runtime->machine_manager()->push_machine(std::move(result));
}

MachineTracker::~MachineTracker()
{
  auto* runtime = Runtime::get_runtime();
  runtime->machine_manager()->pop_machine();
}

const mapping::MachineDesc& MachineTracker::get_current_machine() const
{
  return Runtime::get_runtime()->get_machine();
}

}  // namespace legate
