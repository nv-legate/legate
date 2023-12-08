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

#pragma once

#include "core/mapping/mapping.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/shared_ptr.h"

#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <vector>

/**
 * @file
 * @brief Legate machine interface
 */

namespace legate::mapping {

/**
 * @ingroup mapping
 * @brief A class to represent a range of nodes.
 *
 * `NodeRange`s are half-open intervals of logical node IDs.
 */
struct NodeRange {
  uint32_t low{0};
  uint32_t high{0};

  bool operator<(const NodeRange& other) const noexcept;
  bool operator==(const NodeRange& other) const noexcept;
  bool operator!=(const NodeRange& other) const noexcept;
};

/**
 * @ingroup mapping
 * @brief A class to represent a range of processors.
 *
 * `ProcessorRange`s are half-open intervals of logical processors IDs.
 */
struct ProcessorRange {
  /**
   * @brief Starting processor ID
   */
  uint32_t low{0};
  /**
   * @brief End processor ID
   */
  uint32_t high{0};
  /**
   * @brief Number of per-node processors
   */
  uint32_t per_node_count{1};
  /**
   * @brief Returns the number of processors in the range
   *
   * @return Processor count
   */
  [[nodiscard]] uint32_t count() const noexcept;
  /**
   * @brief Checks if the processor range is empty
   *
   * @return true The range is empty
   * @return false The range is not empty
   */
  [[nodiscard]] bool empty() const noexcept;
  /**
   * @brief Slices the processor range for a given sub-range
   *
   * @param from Starting index
   * @param to End index
   *
   * @return Sliced procesor range
   */
  [[nodiscard]] ProcessorRange slice(uint32_t from, uint32_t to) const;
  /**
   * @brief Computes a range of node IDs for this processor range
   *
   * @return Node range in a pair
   */
  [[nodiscard]] NodeRange get_node_range() const;
  /**
   * @brief Converts the range to a human-readable string
   *
   * @return Processor range in a string
   */
  [[nodiscard]] std::string to_string() const;
  /**
   * @brief Creates an empty processor range
   */
  ProcessorRange() = default;
  /**
   * @brief Creates a processor range
   *
   * @param low_id Starting processor ID
   * @param high_id End processor ID
   * @param per_node_proc_count Number of per-node processors
   */
  ProcessorRange(uint32_t low_id, uint32_t high_id, uint32_t per_node_proc_count) noexcept;

  ProcessorRange operator&(const ProcessorRange&) const;
  bool operator==(const ProcessorRange& other) const noexcept;
  bool operator!=(const ProcessorRange& other) const noexcept;
  bool operator<(const ProcessorRange& other) const noexcept;
  [[nodiscard]] size_t hash() const noexcept;
};

std::ostream& operator<<(std::ostream& stream, const ProcessorRange& range);

namespace detail {
struct Machine;
}  // namespace detail

/**
 * @ingroup mapping
 * @brief Machine descriptor class
 *
 * A `Machine` object describes the machine resource that should be used for a given scope of
 * execution. By default, the scope is given the entire machine resource configured for this
 * process. Then, the client can limit the resource by extracting a portion of the machine
 * and setting it for the scope using `MachineTracker`. Configuring the scope with an
 * empty machine raises a `std::runtime_error` exception.
 */
class Machine {
 public:
  /**
   * @brief Preferred processor type of this machine descriptor
   */
  [[nodiscard]] TaskTarget preferred_target() const;
  /**
   * @brief Returns the processor range for the preferred processor type in this descriptor
   *
   * @return A processor range
  ` */
  [[nodiscard]] ProcessorRange processor_range() const;
  /**
   * @brief Returns the processor range for a given processor type
   *
   * If the processor type does not exist in the descriptor, an empty range is returned
   *
   * @param target Processor type to query
   *
   * @return A processor range
   */
  [[nodiscard]] ProcessorRange processor_range(TaskTarget target) const;
  /**
   * @brief Returns the valid task targets within this machine descriptor
   *
   * @return Task targets
   */
  [[nodiscard]] std::vector<TaskTarget> valid_targets() const;
  /**
   * @brief Returns the valid task targets excluding a given set of targets
   *
   * @param to_exclude Task targets to exclude from the query
   *
   * @return Task targets
   */
  [[nodiscard]] std::vector<TaskTarget> valid_targets_except(
    const std::set<TaskTarget>& to_exclude) const;
  /**
   * @brief Returns the number of preferred processors
   *
   * @return Processor count
   */
  [[nodiscard]] uint32_t count() const;
  /**
   * @brief Returns the number of processors of a given type
   *
   * @param target Processor type to query
   *
   * @return Processor count
   */
  [[nodiscard]] uint32_t count(TaskTarget target) const;

  /**
   * @brief Converts the machine descriptor to a human-readable string
   *
   * @return Machine descriptor in a string
   */
  [[nodiscard]] std::string to_string() const;
  /**
   * @brief Extracts the processor range for a given processor type and creates a fresh machine
   * descriptor with it
   *
   * If the `target` does not exist in the machine descriptor, an empty descriptor is returned.
   *
   * @param target Processor type to select
   *
   * @return Machine descriptor with the chosen processor range
   */
  [[nodiscard]] Machine only(TaskTarget target) const;
  /**
   * @brief Extracts the processor ranges for a given set of processor types and creates a fresh
   * machine descriptor with them
   *
   * Any of the `targets` that does not exist will be mapped to an empty processor range in the
   * returned machine descriptor
   *
   * @param targets Processor types to select
   *
   * @return Machine descriptor with the chosen processor ranges
   */
  [[nodiscard]] Machine only(const std::vector<TaskTarget>& targets) const;
  /**
   * @brief Slices the processor range for a given processor type
   *
   * @param from Starting index
   * @param to End index
   * @param target Processor type to slice
   * @param keep_others Optional flag to keep unsliced ranges in the returned machine descriptor
   *
   * @return Machine descriptor with the chosen procssor range sliced
   */
  [[nodiscard]] Machine slice(uint32_t from,
                              uint32_t to,
                              TaskTarget target,
                              bool keep_others = false) const;
  /**
   * @brief Slices the processor range for the preferred processor type of this machine descriptor
   *
   * @param from Starting index
   * @param to End index
   * @param keep_others Optional flag to keep unsliced ranges in the returned machine descriptor
   *
   * @return Machine descriptor with the preferred processor range sliced
   */
  [[nodiscard]] Machine slice(uint32_t from, uint32_t to, bool keep_others = false) const;
  /**
   * @brief Selects the processor range for a given processor type and constructs a machine
   * descriptor with it.
   *
   * This yields the same result as `.only(target)`.
   *
   * @param target Processor type to select
   *
   * @return Machine descriptor with the chosen processor range
   */
  Machine operator[](TaskTarget target) const;
  /**
   * @brief Selects the processor ranges for a given set of processor types and constructs a machine
   * descriptor with them.
   *
   * This yields the same result as `.only(targets)`.
   *
   * @param targets Processor types to select
   *
   * @return Machine descriptor with the chosen processor ranges
   */
  Machine operator[](const std::vector<TaskTarget>& targets) const;
  bool operator==(const Machine& other) const;
  bool operator!=(const Machine& other) const;
  /**
   * @brief Computes an intersection between two machine descriptors
   *
   * @param other Machine descriptor to intersect with this descriptor
   *
   * @return Machine descriptor
   */
  Machine operator&(const Machine& other) const;
  /**
   * @brief Indicates whether the machine descriptor is empty.
   *
   * A machine descriptor is empty when all its processor ranges are empty
   *
   * @return true The machine descriptor is empty
   * @return false The machine descriptor is non-empty
   */
  [[nodiscard]] bool empty() const;

  Machine() = default;
  explicit Machine(std::map<TaskTarget, ProcessorRange> ranges);
  explicit Machine(InternalSharedPtr<detail::Machine> impl);
  explicit Machine(detail::Machine impl);

  Machine(const Machine&)                = default;
  Machine& operator=(const Machine&)     = default;
  Machine(Machine&&) noexcept            = default;
  Machine& operator=(Machine&&) noexcept = default;

  [[nodiscard]] const SharedPtr<detail::Machine>& impl() const;

 private:
  SharedPtr<detail::Machine> impl_{};
};

std::ostream& operator<<(std::ostream& stream, const Machine& machine);

}  // namespace legate::mapping

#include "core/mapping/machine.inl"
