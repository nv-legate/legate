/* Copyright 2022 NVIDIA Corporation
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

#pragma once

#include <tuple>

#include "core/mapping/mapping.h"
#include "core/utilities/span.h"
#include "legate_defines.h"
#include "legion.h"

/**
 * @file
 * @brief Legate machine interface
 */

namespace legate::mapping {

TaskTarget to_target(Processor::Kind kind);

Processor::Kind to_kind(TaskTarget target);

LegateVariantCode to_variant_code(TaskTarget target);

/**
 * @ingroup mapping
 * @brief A class to represent a range of processors.
 *
 * `ProcessorRange`s are half-open intervals of logical processors IDs.
 */
struct ProcessorRange {
  /**
   * @brief Creates an empty processor range
   */
  ProcessorRange() {}
  /**
   * @brief Creates a processor range
   *
   * @param low Starting processor ID
   * @param high End processor ID
   * @param per_node_count Number of per-node processors
   */
  ProcessorRange(uint32_t low, uint32_t high, uint32_t per_node_count);
  ProcessorRange operator&(const ProcessorRange&) const;
  bool operator==(const ProcessorRange&) const;
  bool operator!=(const ProcessorRange&) const;
  bool operator<(const ProcessorRange&) const;
  /**
   * @brief Returns the number of processors in the range
   *
   * @return Processor count
   */
  uint32_t count() const;
  /**
   * @brief Checks if the processor range is empty
   *
   * @return true The range is empty
   * @return false The range is not empty
   */
  bool empty() const;
  /**
   * @brief Converts the range to a human-readable string
   *
   * @return Processor range in a string
   */
  std::string to_string() const;
  /**
   * @brief Computes a range of node IDs for this processor range
   *
   * @return Node range in a pair
   */
  std::pair<uint32_t, uint32_t> get_node_range() const;
  /**
   * @brief Slices the processor range for a given sub-range
   *
   * @param from Starting index
   * @param to End index
   *
   * @return Sliced procesor range
   */
  ProcessorRange slice(uint32_t from, uint32_t to) const;
  /**
   * @brief Serializes the processor range to a buffer
   *
   * @param buffer Buffer to which the processor range should be serialized
   */
  void pack(BufferBuilder& buffer) const;

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
};

/**
 * @ingroup mapping
 * @brief Machine descriptor class
 *
 * A machine descriptor describes the machine resource that should be used for a given scope of
 * execution. By default, the scope is given the entire machine resource configured for this
 * process. Then, the client can limit the resource by extracting a portion of the machine
 * descriptor and setting it for the scope using `MachineTracker`. Configuring the scope with an
 * empty machine descriptor raises a `std::runtime_error` exception.
 */
struct MachineDesc {
  /**
   * @brief Creates an empty machine descriptor
   */
  MachineDesc() {}
  /**
   * @brief Creates a machine descriptor with a given set of processor ranges
   *
   * @param processor_ranges Processor ranges
   */
  MachineDesc(const std::map<TaskTarget, ProcessorRange>& processor_ranges);
  /**
   * @brief Creates a machine descriptor with a given set of processor ranges
   *
   * @param processor_ranges Processor ranges
   */
  MachineDesc(std::map<TaskTarget, ProcessorRange>&& processor_ranges);

  MachineDesc(const MachineDesc&)            = default;
  MachineDesc& operator=(const MachineDesc&) = default;

  MachineDesc(MachineDesc&&)            = default;
  MachineDesc& operator=(MachineDesc&&) = default;

  /**
   * @brief Returns the processor range for the preferred processor type in this descriptor
   */
  const ProcessorRange& processor_range() const;
  /**
   * @brief Returns the processor range for a given processor type
   *
   * If the processor type does not exist in the descriptor, an empty range is returned
   *
   * @target target Processor type to query
   */
  const ProcessorRange& processor_range(TaskTarget target) const;

  /**
   * @brief Returns the valid task targets within this machine descriptor
   *
   * @return Task targets
   */
  std::vector<TaskTarget> valid_targets() const;
  /**
   * @brief Returns the valid task targets excluding a given set of targets
   *
   * @param to_exclude Task targets to exclude from the query
   *
   * @return Task targets
   */
  std::vector<TaskTarget> valid_targets_except(const std::set<TaskTarget>& to_exclude) const;

  /**
   * @brief Returns the number of preferred processors
   *
   * @return Processor count
   */
  uint32_t count() const;
  /**
   * @brief Returns the number of processors of a given type
   *
   * @param target Processor type to query
   *
   * @return Processor count
   */
  uint32_t count(TaskTarget target) const;

  /**
   * @brief Converts the machine descriptor to a human-readable string
   *
   * @return Machine descriptor in a string
   */
  std::string to_string() const;

  /**
   * @brief Serializes the machine descriptor to a buffer
   *
   * @param buffer Buffer to which the machine descriptor should be serialized
   */
  void pack(BufferBuilder& buffer) const;

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
  MachineDesc only(TaskTarget target) const;
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
  MachineDesc only(const std::vector<TaskTarget>& targets) const;
  /**
   * @brief Slices the processor range for a given processor type
   *
   * @param from Starting index
   * @param to End index
   * @param targets Processor type to slice
   * @param keep_others Optional flag to keep unsliced ranges in the returned machine descriptor
   *
   * @return Machine descriptor with the chosen procssor range sliced
   */
  MachineDesc slice(uint32_t from, uint32_t to, TaskTarget target, bool keep_others = false) const;
  /**
   * @brief Slices the processor range for the preferred processor type of this machine descriptor
   *
   * @param from Starting index
   * @param to End index
   * @param keep_others Optional flag to keep unsliced ranges in the returned machine descriptor
   *
   * @return Machine descriptor with the preferred processor range sliced
   */
  MachineDesc slice(uint32_t from, uint32_t to, bool keep_others = false) const;

  /**
   * @brief Selects the processor range for a given processor type and constructs a machine
   * descriptor with it.
   *
   * This yields the same result as `.only(target)`.
   *
   * @param targets Processor type to select
   *
   * @return Machine descriptor with the chosen processor range
   */
  MachineDesc operator[](TaskTarget target) const;
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
  MachineDesc operator[](const std::vector<TaskTarget>& targets) const;
  bool operator==(const MachineDesc& other) const;
  bool operator!=(const MachineDesc& other) const;
  /**
   * @brief Computes an intersection between two machine descriptors
   *
   * @param other Machine descriptor to intersect with this descriptor
   *
   * @return Machine descriptor
   */
  MachineDesc operator&(const MachineDesc& other) const;

  /**
   * @brief Indicates whether the machine descriptor is empty.
   *
   * A machine descriptor is empty when all its processor ranges are empty
   *
   * @return true The machine descriptor is empty
   * @return false The machine descriptor is non-empty
   */
  bool empty() const;

  /**
   * @brief Preferred processor type of this machine descriptor
   */
  TaskTarget preferred_target{TaskTarget::CPU};
  /**
   * @brief Processor ranges in this machine descriptor
   */
  std::map<TaskTarget, ProcessorRange> processor_ranges{};
};

std::ostream& operator<<(std::ostream& stream, const MachineDesc& info);

class Machine;

class LocalProcessorRange {
 private:
  friend class Machine;
  LocalProcessorRange();
  LocalProcessorRange(const std::vector<Processor>& procs);
  LocalProcessorRange(uint32_t offset,
                      uint32_t total_proc_count,
                      const Processor* local_procs,
                      size_t num_local_procs);

 public:
  const Processor& first() const { return *procs_.begin(); }
  const Processor& operator[](uint32_t idx) const;

 public:
  bool empty() const { return procs_.size() == 0; }

 private:
  uint32_t offset_;
  uint32_t total_proc_count_;
  Span<const Processor> procs_;
};

class Machine {
 public:
  Machine(Legion::Machine legion_machine);

 public:
  const std::vector<Processor>& cpus() const { return cpus_; }
  const std::vector<Processor>& gpus() const { return gpus_; }
  const std::vector<Processor>& omps() const { return omps_; }
  const std::vector<Processor>& procs(TaskTarget target) const;

 public:
  size_t total_cpu_count() const { return total_nodes * cpus_.size(); }
  size_t total_gpu_count() const { return total_nodes * gpus_.size(); }
  size_t total_omp_count() const { return total_nodes * omps_.size(); }

 public:
  size_t total_frame_buffer_size() const;
  size_t total_socket_memory_size() const;

 public:
  bool has_cpus() const { return !cpus_.empty(); }
  bool has_gpus() const { return !gpus_.empty(); }
  bool has_omps() const { return !omps_.empty(); }

 public:
  bool has_socket_memory() const;

 public:
  Memory get_memory(Processor proc, StoreTarget target) const;
  Memory system_memory() const { return system_memory_; }
  Memory zerocopy_memory() const { return zerocopy_memory_; }
  const std::map<Processor, Memory>& frame_buffers() const { return frame_buffers_; }
  const std::map<Processor, Memory>& socket_memories() const { return socket_memories_; }

 public:
  LocalProcessorRange slice(TaskTarget target,
                            const MachineDesc& machine_desc,
                            bool fallback_to_global = false) const;

 public:
  const uint32_t local_node;
  const uint32_t total_nodes;

 private:
  std::vector<Processor> cpus_;
  std::vector<Processor> gpus_;
  std::vector<Processor> omps_;

 private:
  Memory system_memory_, zerocopy_memory_;
  std::map<Processor, Memory> frame_buffers_;
  std::map<Processor, Memory> socket_memories_;
};

}  // namespace legate::mapping
