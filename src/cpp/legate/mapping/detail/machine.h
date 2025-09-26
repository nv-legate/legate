/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/machine.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace legate::mapping::detail {

class Machine {
 public:
  Machine() = default;
  explicit Machine(std::map<TaskTarget, ProcessorRange> processor_ranges);
  Machine(TaskTarget preferred_target, std::map<TaskTarget, ProcessorRange> processor_ranges);

  [[nodiscard]] const ProcessorRange& processor_range() const;
  [[nodiscard]] const ProcessorRange& processor_range(TaskTarget target) const;

  [[nodiscard]] Span<const TaskTarget> valid_targets() const;
  [[nodiscard]] legate::detail::SmallVector<TaskTarget, NUM_TASK_TARGETS> valid_targets_except(
    const std::set<TaskTarget>& to_exclude) const;

  [[nodiscard]] std::uint32_t count() const;
  [[nodiscard]] std::uint32_t count(TaskTarget target) const;

  [[nodiscard]] std::string to_string() const;

  void pack(legate::detail::BufferBuilder& buffer) const;

  template <typename F>
  [[nodiscard]] Machine only_if(F&& pred) const;

  [[nodiscard]] Machine only(TaskTarget target) const;
  [[nodiscard]] Machine only(Span<const TaskTarget> targets) const;
  [[nodiscard]] Machine slice(std::uint32_t from,
                              std::uint32_t to,
                              TaskTarget target,
                              bool keep_others = false) const;
  [[nodiscard]] Machine slice(std::uint32_t from, std::uint32_t to, bool keep_others = false) const;

  [[nodiscard]] Machine operator[](TaskTarget target) const;
  [[nodiscard]] Machine operator[](Span<const TaskTarget> targets) const;
  bool operator==(const Machine& other) const;
  bool operator!=(const Machine& other) const;
  [[nodiscard]] Machine operator&(const Machine& other) const;

  [[nodiscard]] bool empty() const;

  [[nodiscard]] TaskTarget preferred_target() const;
  [[nodiscard]] VariantCode preferred_variant() const;
  [[nodiscard]] const std::map<TaskTarget, ProcessorRange>& processor_ranges() const;

 private:
  TaskTarget preferred_target_{TaskTarget::CPU};
  std::map<TaskTarget, ProcessorRange> processor_ranges_{};
  mutable std::optional<legate::detail::SmallVector<TaskTarget, NUM_TASK_TARGETS>> valid_targets_{};
};

std::ostream& operator<<(std::ostream& os, const Machine& machine);

class LocalProcessorRange {
 public:
  LocalProcessorRange() = default;
  LocalProcessorRange(std::uint32_t offset,
                      std::uint32_t total_proc_count,
                      const Processor* local_procs,
                      std::size_t num_local_procs);

  explicit LocalProcessorRange(const std::vector<Processor>& procs);

  [[nodiscard]] const Processor& first() const;
  [[nodiscard]] const Processor& operator[](std::uint32_t idx) const;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::uint32_t total_proc_count() const;

  friend std::ostream& operator<<(std::ostream& os, const LocalProcessorRange& range);

 private:
  std::uint32_t offset_{};
  std::uint32_t total_proc_count_{};
  Span<const Processor> procs_{};
};

// A machine object holding handles to local processors and memories
class LocalMachine {
 public:
  LocalMachine();

  [[nodiscard]] const std::vector<Processor>& cpus() const;
  [[nodiscard]] const std::vector<Processor>& gpus() const;
  [[nodiscard]] const std::vector<Processor>& omps() const;
  [[nodiscard]] const std::vector<Processor>& procs(TaskTarget target) const;

  [[nodiscard]] std::size_t total_cpu_count() const;
  [[nodiscard]] std::size_t total_gpu_count() const;
  [[nodiscard]] std::size_t total_omp_count() const;

  [[nodiscard]] std::size_t total_frame_buffer_size() const;
  [[nodiscard]] std::size_t total_socket_memory_size() const;
  [[nodiscard]] std::size_t total_system_memory_size() const;

  /**
   * @brief Compute how many bytes of allocations should occur before we trigger a consensus
   * match.
   *
   * @param field_reuse_frac What fraction of memory should be allocated before consensus match
   * is triggered.
   *
   * @return The consensus match allocation frequency, in bytes of intervening allocations.
   */
  [[nodiscard]] std::size_t calculate_field_reuse_size(std::uint32_t field_reuse_frac) const;

  [[nodiscard]] bool has_cpus() const;
  [[nodiscard]] bool has_gpus() const;
  [[nodiscard]] bool has_omps() const;

  [[nodiscard]] bool has_socket_memory() const;

  [[nodiscard]] const Processor& find_first_processor_with_affinity_to(StoreTarget target) const;
  [[nodiscard]] Memory get_memory(Processor proc, StoreTarget target) const;
  /**
   * @brief Retrieve the memory handle for a particular kind of memory on a processor.
   *
   * @param proc The processor on which the memory should reside.
   * @param kind The kind of memory to retrieve.
   *
   * @return The memory handle.
   *
   * @throw std::invalid_argument If the memory kind is invalid.
   * @throw std::out_of_range If the processor does not support the requested memory kind.
   */
  [[nodiscard]] Memory get_memory(Processor proc, Memory::Kind kind) const;
  [[nodiscard]] Memory system_memory() const;
  [[nodiscard]] Memory zerocopy_memory() const;
  [[nodiscard]] const std::map<Processor, Memory>& frame_buffers() const;
  [[nodiscard]] const std::map<Processor, Memory>& socket_memories() const;

  [[nodiscard]] std::uint32_t g2c_multi_hop_bandwidth(Memory gpu_mem, Memory cpu_mem) const;

  [[nodiscard]] LocalProcessorRange slice(TaskTarget target,
                                          const Machine& machine,
                                          bool fallback_to_global = false) const;

  std::uint32_t node_id{};
  std::uint32_t total_nodes{};

 private:
  void init_g2c_multi_hop_bandwidth_();

  std::vector<Processor> cpus_{};
  std::vector<Processor> gpus_{};
  std::vector<Processor> omps_{};

  Memory system_memory_, zerocopy_memory_;
  std::map<Processor, Memory> frame_buffers_{};
  std::map<Processor, Memory> socket_memories_{};
  std::unordered_map<Memory, std::unordered_map<Memory, std::uint32_t>> g2c_multi_hop_bandwidth_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/machine.inl>
