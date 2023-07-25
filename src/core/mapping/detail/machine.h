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

#pragma once

#include "core/mapping/detail/mapping.h"
#include "core/mapping/machine.h"
#include "core/mapping/mapping.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/typedefs.h"

namespace legate::mapping::detail {

struct Machine {
  Machine() {}
  Machine(const std::map<TaskTarget, ProcessorRange>& processor_ranges);
  Machine(std::map<TaskTarget, ProcessorRange>&& processor_ranges);

  Machine(const Machine&)            = default;
  Machine& operator=(const Machine&) = default;

  Machine(Machine&&)            = default;
  Machine& operator=(Machine&&) = default;

  const ProcessorRange& processor_range() const;
  const ProcessorRange& processor_range(TaskTarget target) const;

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
};

std::ostream& operator<<(std::ostream& stream, const Machine& machine);

class LocalProcessorRange {
 public:
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

 public:
  std::string to_string() const;

 private:
  uint32_t offset_;
  uint32_t total_proc_count_;
  Span<const Processor> procs_;
};

std::ostream& operator<<(std::ostream& stream, const LocalProcessorRange& info);

// A machine object holding handles to local processors and memories
class LocalMachine {
 public:
  LocalMachine();

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
                            const Machine& machine,
                            bool fallback_to_global = false) const;

 public:
  const uint32_t node_id;
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

}  // namespace legate::mapping::detail
