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

#include "core/mapping/machine.h"
#include "core/mapping/mapping.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/typedefs.h"

#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace legate::mapping::detail {

struct Machine {
  Machine() = default;
  explicit Machine(const std::map<TaskTarget, ProcessorRange>& processor_ranges);
  explicit Machine(std::map<TaskTarget, ProcessorRange>&& processor_ranges);

  [[nodiscard]] const ProcessorRange& processor_range() const;
  [[nodiscard]] const ProcessorRange& processor_range(TaskTarget target) const;

  std::vector<TaskTarget> valid_targets() const;
  std::vector<TaskTarget> valid_targets_except(const std::set<TaskTarget>& to_exclude) const;

  uint32_t count() const;
  uint32_t count(TaskTarget target) const;

  std::string to_string() const;

  void pack(legate::detail::BufferBuilder& buffer) const;

  Machine only(TaskTarget target) const;
  Machine only(const std::vector<TaskTarget>& targets) const;
  Machine slice(uint32_t from, uint32_t to, TaskTarget target, bool keep_others = false) const;
  Machine slice(uint32_t from, uint32_t to, bool keep_others = false) const;

  Machine operator[](TaskTarget target) const;
  Machine operator[](const std::vector<TaskTarget>& targets) const;
  bool operator==(const Machine& other) const;
  bool operator!=(const Machine& other) const;
  Machine operator&(const Machine& other) const;

  bool empty() const;

  TaskTarget preferred_target{TaskTarget::CPU};
  std::map<TaskTarget, ProcessorRange> processor_ranges{};

 private:
  struct private_tag {};

  explicit Machine(private_tag, std::map<TaskTarget, ProcessorRange> processor_ranges);
};

std::ostream& operator<<(std::ostream& stream, const Machine& machine);

class LocalProcessorRange {
 public:
  LocalProcessorRange() = default;
  LocalProcessorRange(uint32_t offset,
                      uint32_t total_proc_count,
                      const Processor* local_procs,
                      size_t num_local_procs);

  explicit LocalProcessorRange(const std::vector<Processor>& procs);

  [[nodiscard]] const Processor& first() const;
  [[nodiscard]] const Processor& operator[](uint32_t idx) const;

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] uint32_t total_proc_count() const;

 private:
  uint32_t offset_{};
  uint32_t total_proc_count_{};
  Span<const Processor> procs_{};
};

std::ostream& operator<<(std::ostream& stream, const LocalProcessorRange& range);

// A machine object holding handles to local processors and memories
class LocalMachine {
 public:
  LocalMachine();

  [[nodiscard]] const std::vector<Processor>& cpus() const;
  [[nodiscard]] const std::vector<Processor>& gpus() const;
  [[nodiscard]] const std::vector<Processor>& omps() const;
  [[nodiscard]] const std::vector<Processor>& procs(TaskTarget target) const;

  [[nodiscard]] size_t total_cpu_count() const;
  [[nodiscard]] size_t total_gpu_count() const;
  [[nodiscard]] size_t total_omp_count() const;

  [[nodiscard]] size_t total_frame_buffer_size() const;
  [[nodiscard]] size_t total_socket_memory_size() const;

  [[nodiscard]] bool has_cpus() const;
  [[nodiscard]] bool has_gpus() const;
  [[nodiscard]] bool has_omps() const;

  [[nodiscard]] bool has_socket_memory() const;

  [[nodiscard]] Memory get_memory(Processor proc, StoreTarget target) const;
  [[nodiscard]] Memory system_memory() const;
  [[nodiscard]] Memory zerocopy_memory() const;
  [[nodiscard]] const std::map<Processor, Memory>& frame_buffers() const;
  [[nodiscard]] const std::map<Processor, Memory>& socket_memories() const;

  [[nodiscard]] LocalProcessorRange slice(TaskTarget target,
                                          const Machine& machine,
                                          bool fallback_to_global = false) const;

  uint32_t node_id{};
  uint32_t total_nodes{};

 private:
  std::vector<Processor> cpus_{};
  std::vector<Processor> gpus_{};
  std::vector<Processor> omps_{};

  Memory system_memory_, zerocopy_memory_;
  std::map<Processor, Memory> frame_buffers_{};
  std::map<Processor, Memory> socket_memories_{};
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/machine.inl"
