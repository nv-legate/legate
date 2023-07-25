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

#include <string>

#include "core/mapping/machine.h"

/**
 * @file
 * @brief Class definitions for legate::ProvenanceTracker and legate::MachineTracker
 */

namespace legate {

/**
 * @ingroup util
 * @brief A helper class to set provenance information for the scope
 *
 * Client programs often want to attach provenance information to each of their operations and hvae
 * it rendered in profiling outputs. In such scenarios, client programs can create a
 * `ProvenanceTracker` object with the provenance string in a scope and issue operations. All the
 * issued operations then will be associated with the provenance information, which will be attached
 * to them in profiling outputs.
 */
struct ProvenanceTracker {
  /**
   * @brief Creates a trakcer that sets a given provenance string for the scope
   *
   * @param provenance Provenance information in string
   */
  ProvenanceTracker(const std::string& provenance);

  /**
   * @brief Pops out the provenance string set by this tracker
   */
  ~ProvenanceTracker();

  /**
   * @brief Returns the current provenance string
   *
   * @return Provenance string
   */
  const std::string& get_current_provenance() const;
};

/**
 * @ingroup util
 * @brief A helper class to configure machine for the scope
 *
 * By default, Legate operations target the entire machine available for the program. When a client
 * program wants to assign a subset of the machine to its operations, it can subdivide the machine
 * using the machine API (see `Machine` for details) and set a sub-machine for the scope using
 * `MachineTracker`. All operations within the scope where the `MachineTracker` object is alive will
 * only target that sub-machine, instead of the entire machine.
 *
 */
struct MachineTracker {
  /**
   * @brief Creates a trakcer that sets a given machine for the scope
   *
   * @param machine Machine to use for the scope
   */
  MachineTracker(const mapping::Machine& machine);

  /**
   * @brief Pops out the machine set by this tracker
   */
  ~MachineTracker();

  /**
   * @brief Returns the current machine
   *
   * @return Machine
   */
  mapping::Machine get_current_machine() const;
};

}  // namespace legate
