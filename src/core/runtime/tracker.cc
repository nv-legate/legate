/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/runtime/tracker.h"

#include "core/mapping/detail/machine.h"
#include "core/runtime/detail/machine_manager.h"
#include "core/runtime/detail/provenance_manager.h"
#include "core/runtime/detail/runtime.h"

namespace legate {

//////////////////////////////////////
//  ProvenanceTracker
//////////////////////////////////////

ProvenanceTracker::ProvenanceTracker(const std::string& p)
{
  auto* runtime = detail::Runtime::get_runtime();
  runtime->provenance_manager()->push_provenance(p);
}

ProvenanceTracker::~ProvenanceTracker()
{
  auto* runtime = detail::Runtime::get_runtime();
  runtime->provenance_manager()->pop_provenance();
}

const std::string& ProvenanceTracker::get_current_provenance() const
{
  return detail::Runtime::get_runtime()->provenance_manager()->get_provenance();
}

////////////////////////////////////////////
// legate::MachineTracker
////////////////////////////////////////////

MachineTracker::MachineTracker(const mapping::Machine& machine)
{
  auto* runtime = detail::Runtime::get_runtime();
  auto result   = machine & mapping::Machine(runtime->get_machine());
  if (result.count() == 0)
    throw std::runtime_error("Empty machines cannot be used for resource scoping");
  runtime->machine_manager()->push_machine(std::move(*result.impl()));
}

MachineTracker::~MachineTracker()
{
  auto* runtime = detail::Runtime::get_runtime();
  runtime->machine_manager()->pop_machine();
}

mapping::Machine MachineTracker::get_current_machine() const
{
  return mapping::Machine(detail::Runtime::get_runtime()->get_machine());
}

}  // namespace legate
