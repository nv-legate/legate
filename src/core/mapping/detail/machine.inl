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

#include "core/mapping/detail/machine.h"

namespace legate::mapping::detail {

inline Machine Machine::operator[](TaskTarget target) const { return only(target); }

inline Machine Machine::operator[](const std::vector<TaskTarget>& targets) const
{
  return only(targets);
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

inline const Processor& LocalProcessorRange::first() const { return *procs_.begin(); }

inline bool LocalProcessorRange::empty() const { return procs_.size() == 0; }

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
