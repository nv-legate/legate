/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>

namespace legate::mapping::detail {

inline TaskTarget Machine::preferred_target() const { return preferred_target_; }

inline const std::map<TaskTarget, ProcessorRange>& Machine::processor_ranges() const
{
  return processor_ranges_;
}

template <typename F>
Machine Machine::only_if(F&& pred) const
{
  std::map<TaskTarget, ProcessorRange> new_processor_ranges;

  static_assert(std::is_invocable_r_v<bool, F, TaskTarget>);
  for (auto&& t : valid_targets()) {
    if (pred(t)) {
      new_processor_ranges.insert({t, processor_range(t)});
    }
  }

  return Machine{std::move(new_processor_ranges)};
}

// ==========================================================================================

inline LocalProcessorRange::LocalProcessorRange(std::uint32_t offset,
                                                std::uint32_t total_proc_count,
                                                const Processor* local_procs,
                                                std::size_t num_local_procs)
  : offset_{offset}, total_proc_count_{total_proc_count}, procs_{local_procs, num_local_procs}
{
}

inline LocalProcessorRange::LocalProcessorRange(const std::vector<Processor>& procs)
  : LocalProcessorRange{0, static_cast<std::uint32_t>(procs.size()), procs.data(), procs.size()}
{
}

inline const Processor& LocalProcessorRange::first() const { return procs_.front(); }

inline bool LocalProcessorRange::empty() const { return procs_.empty(); }

inline std::uint32_t LocalProcessorRange::total_proc_count() const { return total_proc_count_; }

// ==========================================================================================

inline const std::vector<Processor>& LocalMachine::cpus() const { return cpus_; }

inline const std::vector<Processor>& LocalMachine::gpus() const { return gpus_; }

inline const std::vector<Processor>& LocalMachine::omps() const { return omps_; }

inline std::size_t LocalMachine::total_cpu_count() const { return total_nodes * cpus().size(); }

inline std::size_t LocalMachine::total_gpu_count() const { return total_nodes * gpus().size(); }

inline std::size_t LocalMachine::total_omp_count() const { return total_nodes * omps().size(); }

inline bool LocalMachine::has_cpus() const { return !cpus_.empty(); }

inline bool LocalMachine::has_gpus() const { return !gpus_.empty(); }

inline bool LocalMachine::has_omps() const { return !omps_.empty(); }

inline Memory LocalMachine::system_memory() const { return system_memory_; }

inline Memory LocalMachine::zerocopy_memory() const { return zerocopy_memory_; }

inline const std::map<Processor, Memory>& LocalMachine::frame_buffers() const
{
  return frame_buffers_;
}

inline const std::map<Processor, Memory>& LocalMachine::socket_memories() const
{
  return socket_memories_;
}

}  // namespace legate::mapping::detail
