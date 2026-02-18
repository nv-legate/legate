/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <functional>
#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace legate::mapping::detail {

/**
 * @brief A class representing the instantaneous machine that Legate is allowed to work with.
 *
 * This class is usable to constrain mapping decisions to a specific slice or subset of the machine.
 */
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
  /**
   * @copydoc mapping::Machine::only()
   */
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

class ProcessorSpan {
 public:
  /**
   * @brief Constructs an empty ProcessorSpan.
   *
   * @return An empty ProcessorSpan with no processors, no offset, and a processor count of 0.
   */
  ProcessorSpan() = default;
  /**
   * @brief Constructs a ProcessorSpan with the given offset, total processor count, and processors.
   *
   * @param offset The starting index offset for this processor range within the global processor
   * space. See ProcessorSpan::operator[] for indexing into the span.
   * @param total_proc_count The total number of processors that exist in the machine. Not
   * necessarily the same as procs.size().
   * @param procs The processors to include in the ProcessorSpan.
   *
   * @return A ProcessorSpan over processors that are index-able from the offset.
   */
  ProcessorSpan(std::uint32_t offset, std::uint32_t total_proc_count, Span<const Processor> procs);

  /**
   * @brief Constructs a ProcessorSpan with the given processors
   *
   * The resulting ProcessorSpan will have an offset of 0 and total processor count
   * equal to procs.size().
   *
   * @param procs The processors to include in the ProcessorSpan.
   *
   * @return A ProcessorSpan over the given processors.
   */
  explicit ProcessorSpan(legate::Span<const Processor> procs);

  [[nodiscard]] const Processor& first() const;
  /**
   * @brief Returns the processor at the given index relative to the offset.
   *
   * For example, if we had ProcessorSpan span{3, 8, {p0, p1}}. Then, span[3] would return p0,
   * and span[4] would return p1.
   */
  [[nodiscard]] const Processor& operator[](std::uint32_t idx) const;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::string to_string() const;
  /**
   * @brief Returns the total number of processors that exist in the machine slice which is not
   * necessarily the same as the number of processors in span that ProcessorSpan was built with.
   *
   * @return The total number of global processors.
   */
  [[nodiscard]] std::uint32_t total_proc_count() const;
  /**
   * @brief Returns the starting index offset for this processor range within the global processor
   * space.
   *
   * @return The offset value used for indexing processors in the range.
   */
  [[nodiscard]] std::uint32_t offset() const;
  /**
   * @brief Returns the number of processors locally available in this span.
   *
   * @return The local processor count (size of the underlying processor array).
   */
  [[nodiscard]] std::uint32_t local_proc_count() const;

  friend std::ostream& operator<<(std::ostream& os, const ProcessorSpan& range);

 private:
  std::uint32_t offset_{};
  std::uint32_t total_proc_count_{};
  Span<const Processor> procs_{};
};

/**
 * @brief A class that represents some rank of the Legate machine in which all processors and
 * memories are local to each other.
 */
class LocalMachine {
 public:
  /**
   * @brief Constructs a LocalMachine object containing processors and memories owned by the rank.
   *
   * For example, if the machine has 2 ranks, {{CPU0, SYSMEM0}, {CPU1, SYSMEM1}},
   * then LocalMachine(), if called on rank 1 (0-indexed), will contain {CPU1, SYSMEM1}
   * which it owns.
   *
   * @return A LocalMachine object containing processors and memories owned by the calling rank.
   */
  LocalMachine();
  /**
   * @brief Constructs a LocalMachine object representing the same rank as the processor p.
   *
   * For example, if the machine has 2 ranks, {{CPU0, SYSMEM0}, {CPU1, SYSMEM1}},
   * then LocalMachine(CPU1) will contain {CPU1, SYSMEM1}.
   *
   * @param p The processor whose rank will be used to construct the LocalMachine.
   *
   * @return A LocalMachine object containing processors and memories owned by the same rank as p.
   */
  explicit LocalMachine(Processor p);
  /**
   * @brief Constructs a LocalMachine object representing the same rank as the memory m.
   *
   * For example, if the machine has 2 ranks, {{CPU0, SYSMEM0}, {CPU1, SYSMEM1}},
   * then LocalMachine(SYSMEM0) will contain {CPU0, SYSMEM0}.
   *
   * @param m The memory whose rank will be used to construct the LocalMachine.
   *
   * @return A LocalMachine object containing processors and memories owned by the same rank as m.
   */
  explicit LocalMachine(Memory m);

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

  /**
   * @brief Returns a span of processors from the local machine that are within the scope of the
   * machine and its preferred target.
   *
   * @param machine The machine scope to slice processors from its preferred target.
   *
   * @return A span of processors respecting the target type on local machine and within machine
   * slice, if possible. Otherwise, returns an empty span.
   */
  [[nodiscard]] ProcessorSpan slice(const Machine& machine) const;

  /**
   * @brief Returns a span of processors from the local machine that are within the scope of the
   * machine and its preferred target. This method ignores machine if machine does not intersect
   * with the object.
   *
   * @param machine The machine scope to slice processors from its preferred target.
   *
   * @return A span of processors respecting the target type on local machine and within machine
   * slice, if possible. Otherwise, returns all processors on this object for the preferred target.
   */
  [[nodiscard]] ProcessorSpan slice_with_fallback(const Machine& machine) const;

  std::uint32_t node_id{};
  std::uint32_t total_nodes{};

 private:
  /**
   * @brief Constructs an object of the LocalMachine class
   *
   * @param localize_pq Function that localizes a ProcessorQuery to a specific address space and
   * returns it
   * @param localize_mq Function that localizes a MemoryQuery to the same address space as
   * localize_pq and returns it
   */
  LocalMachine(
    const std::function<Legion::Machine::ProcessorQuery&(Legion::Machine::ProcessorQuery&)>&
      localize_pq,
    const std::function<Legion::Machine::MemoryQuery&(Legion::Machine::MemoryQuery&)>& localize_mq);
  /**
   * @brief Initializes the multi-hop bandwidth between GPU and CPU memories
   *
   * @param localize_mq Function that localizes a MemoryQuery to a specific address space
   */
  void init_g2c_multi_hop_bandwidth_(
    const std::function<Legion::Machine::MemoryQuery&(Legion::Machine::MemoryQuery&)>& localize_mq);

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
