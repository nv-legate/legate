/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/mapping/detail/machine.h"

#include "core/utilities/detail/buffer_builder.h"

#include "realm/network.h"

#include <algorithm>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <sstream>
#include <type_traits>
#include <utility>

namespace legate::mapping::detail {

///////////////////////////////////////////
// legate::mapping::detail::Machine
//////////////////////////////////////////

Machine::Machine(std::map<TaskTarget, ProcessorRange> ranges) : processor_ranges_{std::move(ranges)}
{
  for (auto&& [target, processor_range] : processor_ranges()) {
    if (!processor_range.empty()) {
      preferred_target_ = target;
      return;
    }
  }
}

Machine::Machine(TaskTarget preferred_target, std::map<TaskTarget, ProcessorRange> ranges)
  : preferred_target_{preferred_target}, processor_ranges_{std::move(ranges)}
{
}

const ProcessorRange& Machine::processor_range() const
{
  return processor_range(preferred_target());
}

const ProcessorRange& Machine::processor_range(TaskTarget target) const
{
  auto finder = processor_ranges().find(target);
  if (finder == processor_ranges().end()) {
    static constexpr ProcessorRange EMPTY_RANGE{};

    return EMPTY_RANGE;
  }
  return finder->second;
}

const std::vector<TaskTarget>& Machine::valid_targets() const
{
  if (!valid_targets_.has_value()) {
    auto& vec = valid_targets_.emplace();

    vec.reserve(processor_ranges().size());
    for (auto&& [target, range] : processor_ranges()) {
      if (!range.empty()) {
        vec.push_back(target);
      }
    }
  }
  return *valid_targets_;
}

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  std::vector<TaskTarget> result;

  for (auto&& [target, _] : processor_ranges()) {
    if (to_exclude.find(target) == to_exclude.end()) {
      result.push_back(target);
    }
  }
  return result;
}

std::uint32_t Machine::count() const { return count(preferred_target()); }

std::uint32_t Machine::count(TaskTarget target) const { return processor_range(target).count(); }

std::string Machine::to_string() const { return fmt::format("{}", fmt::streamed(*this)); }

void Machine::pack(legate::detail::BufferBuilder& buffer) const
{
  buffer.pack(legate::traits::detail::to_underlying(preferred_target()));
  buffer.pack(static_cast<std::uint32_t>(processor_ranges().size()));
  for (auto&& [target, processor_range] : processor_ranges()) {
    buffer.pack(legate::traits::detail::to_underlying(target));
    buffer.pack<std::uint32_t>(processor_range.low);
    buffer.pack<std::uint32_t>(processor_range.high);
    buffer.pack<std::uint32_t>(processor_range.per_node_count);
  }
}

Machine Machine::only(TaskTarget target) const { return only(std::vector<TaskTarget>{target}); }

Machine Machine::only(const std::vector<TaskTarget>& targets) const
{
  std::map<TaskTarget, ProcessorRange> new_processor_ranges;
  for (auto&& t : targets) {
    new_processor_ranges.insert({t, processor_range(t)});
  }

  return Machine{std::move(new_processor_ranges)};
}

Machine Machine::slice(std::uint32_t from,
                       std::uint32_t to,
                       TaskTarget target,
                       bool keep_others) const
{
  if (keep_others) {
    std::map<TaskTarget, ProcessorRange> new_ranges{processor_ranges()};

    new_ranges[target] = processor_range(target).slice(from, to);
    return Machine{std::move(new_ranges)};
  }
  return Machine{{{target, processor_range(target).slice(from, to)}}};
}

Machine Machine::slice(std::uint32_t from, std::uint32_t to, bool keep_others) const
{
  return slice(from, to, preferred_target(), keep_others);
}

bool Machine::operator==(const Machine& other) const
{
  if (processor_ranges().size() < other.processor_ranges().size()) {
    return other.operator==(*this);
  }
  auto equal_ranges = [&](const auto& proc_range) {
    const auto& [target, range] = proc_range;

    if (range.empty()) {
      return true;
    }
    auto finder = other.processor_ranges().find(target);
    return !(finder == other.processor_ranges().end() || range != finder->second);
  };

  return std::all_of(processor_ranges().begin(), processor_ranges().end(), std::move(equal_ranges));
}

bool Machine::operator!=(const Machine& other) const { return !(*this == other); }

Machine Machine::operator&(const Machine& other) const
{
  std::map<TaskTarget, ProcessorRange> new_processor_ranges;
  for (const auto& [target, range] : processor_ranges()) {
    auto finder = other.processor_ranges().find(target);
    if (finder != other.processor_ranges().end()) {
      new_processor_ranges[target] = finder->second & range;
    }
  }
  return Machine{std::move(new_processor_ranges)};
}

bool Machine::empty() const
{
  return std::all_of(processor_ranges().begin(), processor_ranges().end(), [](auto& rng) {
    return rng.second.empty();
  });
}

std::ostream& operator<<(std::ostream& os, const Machine& machine)
{
  os << "Machine(preferred_target: " << machine.preferred_target();
  for (auto&& [kind, range] : machine.processor_ranges()) {
    os << ", " << kind << ": " << range;
  }
  os << ")";
  return os;
}

///////////////////////////////////////////
// legate::mapping::LocalProcessorRange
///////////////////////////////////////////

const Processor& LocalProcessorRange::operator[](std::uint32_t idx) const
{
  auto local_idx = idx - offset_;
  static_assert(std::is_unsigned_v<decltype(local_idx)>,
                "if local_idx becomes signed, also check local_idx >= 0 below!");
  LEGATE_ASSERT(local_idx < procs_.size());
  return procs_[local_idx];
}

std::string LocalProcessorRange::to_string() const
{
  return fmt::format("{}", fmt::streamed(*this));
}

std::ostream& operator<<(std::ostream& os, const LocalProcessorRange& range)
{
  os << "{offset: " << range.offset_ << ", total processor count: " << range.total_proc_count_
     << ", processors: ";
  for (auto&& proc : range.procs_) {
    os << proc << ",";
  }
  os << "}";
  return os;
}

///////////////////////////////////////////
// legate::mapping::LocalMachine
///////////////////////////////////////////
LocalMachine::LocalMachine()
  : node_id{static_cast<std::uint32_t>(Realm::Network::my_node_id)},
    total_nodes{
      static_cast<std::uint32_t>(Legion::Machine::get_machine().get_address_space_count())}
{
  auto legion_machine = Legion::Machine::get_machine();
  Legion::Machine::ProcessorQuery procs{legion_machine};
  // Query to find all our local processors
  procs.local_address_space();
  for (auto proc : procs) {
    switch (proc.kind()) {
      case Processor::LOC_PROC: {
        cpus_.push_back(proc);
        continue;
      }
      case Processor::TOC_PROC: {
        gpus_.push_back(proc);
        continue;
      }
      case Processor::OMP_PROC: {
        omps_.push_back(proc);
        continue;
      }
      default: {
        continue;
      }
    }
  }

  // Now do queries to find all our local memories
  Legion::Machine::MemoryQuery sysmem{legion_machine};
  sysmem.local_address_space().only_kind(Legion::Memory::SYSTEM_MEM);
  LEGATE_CHECK(sysmem.count() > 0);
  system_memory_ = sysmem.first();

  if (!gpus_.empty()) {
    Legion::Machine::MemoryQuery zcmem{legion_machine};

    zcmem.local_address_space().only_kind(Legion::Memory::Z_COPY_MEM);
    LEGATE_CHECK(zcmem.count() > 0);
    zerocopy_memory_ = zcmem.first();
  }

  for (auto&& gpu : gpus_) {
    Legion::Machine::MemoryQuery framebuffer{legion_machine};

    framebuffer.local_address_space().only_kind(Legion::Memory::GPU_FB_MEM).best_affinity_to(gpu);
    LEGATE_CHECK(framebuffer.count() > 0);
    frame_buffers_[gpu] = framebuffer.first();
  }

  for (auto&& omp : omps_) {
    Legion::Machine::MemoryQuery sockmem{legion_machine};

    sockmem.local_address_space().only_kind(Legion::Memory::SOCKET_MEM).best_affinity_to(omp);
    // If we have socket memories then use them
    if (sockmem.count() > 0) {
      socket_memories_[omp] = sockmem.first();
    }
    // Otherwise we just use the local system memory
    else {
      socket_memories_[omp] = system_memory_;
    }
  }

  init_g2c_multi_hop_bandwidth_();
}

void LocalMachine::init_g2c_multi_hop_bandwidth_()
{
  // Estimate local-node CPU<->GPU multi-hop memory copy bandwidth. If the CPU memory is not pinned,
  // then Realm cannot do this in one hop, and therefore get_mem_mem_affinity won't cover this case.
  // We know Realm will use a GPU-pinned intermediate memory as a bounce buffer, so use the
  // affinities to zerocopy memory as a stand-in for that.
  // TODO(mpapadakis): Will no longer need to do this once Realm provides an API to estimate
  // multi-hop copy bandwidth, see https://github.com/StanfordLegion/legion/issues/1704.
  auto legion_machine = Legion::Machine::get_machine();
  if (zerocopy_memory_.exists()) {
    Legion::Machine::MemoryQuery gpu_mems{legion_machine};

    gpu_mems.local_address_space().only_kind(Legion::Memory::GPU_FB_MEM);
    for (auto gpu_mem : gpu_mems) {
      std::vector<Legion::MemoryMemoryAffinity> g2z_affinities;
      legion_machine.get_mem_mem_affinity(
        g2z_affinities, gpu_mem, zerocopy_memory_, false /*not just local affinities*/);
      if (g2z_affinities.empty()) {
        continue;
      }
      LEGATE_ASSERT(g2z_affinities.size() == 1);
      auto g2z_bandwidth  = g2z_affinities.front().bandwidth;
      auto& cache_for_gpu = g2c_multi_hop_bandwidth_[gpu_mem];

      Legion::Machine::MemoryQuery cpu_mems{legion_machine};
      cpu_mems.local_address_space();
      for (auto cpu_mem : cpu_mems) {
        if (cpu_mem.kind() != Legion::Memory::SYSTEM_MEM &&
            cpu_mem.kind() != Legion::Memory::SOCKET_MEM) {
          continue;
        }
        std::vector<Legion::MemoryMemoryAffinity> c2z_affinities;
        legion_machine.get_mem_mem_affinity(
          c2z_affinities, cpu_mem, zerocopy_memory_, false /*not just local affinities*/);
        if (c2z_affinities.empty()) {
          continue;
        }
        LEGATE_ASSERT(c2z_affinities.size() == 1);
        auto c2z_bandwidth     = c2z_affinities.front().bandwidth;
        cache_for_gpu[cpu_mem] = std::min(g2z_bandwidth, c2z_bandwidth);
      }
    }
  }
}

const std::vector<Processor>& LocalMachine::procs(TaskTarget target) const
{
  switch (target) {
    case TaskTarget::GPU: return gpus_;
    case TaskTarget::OMP: return omps_;
    case TaskTarget::CPU: return cpus_;
  }
  return cpus_;
}

std::size_t LocalMachine::total_frame_buffer_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  const std::size_t per_node_size =
    frame_buffers_.size() * frame_buffers_.begin()->second.capacity();
  return per_node_size * total_nodes;
}

std::size_t LocalMachine::total_socket_memory_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  const std::size_t per_node_size =
    socket_memories_.size() * socket_memories_.begin()->second.capacity();
  return per_node_size * total_nodes;
}

bool LocalMachine::has_socket_memory() const
{
  return !socket_memories_.empty() &&
         socket_memories_.begin()->second.kind() == Legion::Memory::SOCKET_MEM;
}

LocalProcessorRange LocalMachine::slice(TaskTarget target,
                                        const Machine& machine,
                                        bool fallback_to_global /*=false*/) const
{
  const auto& local_procs = procs(target);

  auto finder = machine.processor_ranges().find(target);
  if (machine.processor_ranges().end() == finder) {
    if (fallback_to_global) {
      return LocalProcessorRange{local_procs};
    }
    return {};
  }

  auto& global_range = finder->second;

  auto num_local_procs = local_procs.size();
  auto my_low          = num_local_procs * node_id;
  const ProcessorRange my_range{static_cast<std::uint32_t>(my_low),
                                static_cast<std::uint32_t>(my_low + num_local_procs),
                                global_range.per_node_count};

  auto slice = global_range & my_range;
  if (slice.empty()) {
    if (fallback_to_global) {
      return LocalProcessorRange{local_procs};
    }
    return {};
  }

  return {
    slice.low, global_range.count(), local_procs.data() + (slice.low - my_low), slice.count()};
}

Legion::Memory LocalMachine::get_memory(Processor proc, StoreTarget target) const
{
  switch (target) {
    case StoreTarget::SYSMEM: return system_memory_;
    case StoreTarget::FBMEM: return frame_buffers_.at(proc);
    case StoreTarget::ZCMEM: return zerocopy_memory_;
    case StoreTarget::SOCKETMEM: return socket_memories_.at(proc);
    default: LEGATE_ABORT("invalid StoreTarget: " << legate::traits::detail::to_underlying(target));
  }
  return Legion::Memory::NO_MEMORY;
}

std::uint32_t LocalMachine::g2c_multi_hop_bandwidth(Memory gpu_mem, Memory cpu_mem) const
{
  const auto gpu_finder = g2c_multi_hop_bandwidth_.find(gpu_mem);

  if (gpu_finder == g2c_multi_hop_bandwidth_.end()) {
    return 0;
  }

  const auto cpu_finder = gpu_finder->second.find(cpu_mem);

  if (cpu_finder == gpu_finder->second.end()) {
    return 0;
  }
  return cpu_finder->second;
}

}  // namespace legate::mapping::detail
