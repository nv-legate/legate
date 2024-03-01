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

#include "core/mapping/machine.h"

#include <string>

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
class ProvenanceTracker {
 public:
  /**
   * @brief Creates a tracker that sets a given provenance string for the scope
   *
   * @param provenance Provenance information in string
   */
  explicit ProvenanceTracker(std::string provenance);

  /**
   * @brief Pops out the provenance string set by this tracker
   */
  ~ProvenanceTracker();

  /**
   * @brief Returns the current provenance string
   *
   * @return Provenance string
   */
  [[nodiscard]] static const std::string& get_current_provenance();
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
class MachineTracker {
 public:
  /**
   * @brief Creates a trakcer that sets a given machine for the scope
   *
   * @param machine Machine to use for the scope
   */
  explicit MachineTracker(const mapping::Machine& machine);

  /**
   * @brief Pops out the machine set by this tracker
   */
  ~MachineTracker();

  /**
   * @brief Returns the current machine
   *
   * @return Machine
   */
  [[nodiscard]] static mapping::Machine get_current_machine();
};

}  // namespace legate
