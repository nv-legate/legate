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

#include "core/mapping/detail/machine.h"

#include "core/utilities/detail/buffer_builder.h"

#include "realm/network.h"

#include <algorithm>
#include <sstream>
#include <type_traits>
#include <utility>

namespace legate::mapping::detail {

///////////////////////////////////////////
// legate::mapping::detail::Machine
//////////////////////////////////////////

Machine::Machine(private_tag, std::map<TaskTarget, ProcessorRange> ranges)
  : processor_ranges{std::move(ranges)}
{
  for (auto&& [target, processor_range] : processor_ranges) {
    if (!processor_range.empty()) {
      preferred_target = target;
      return;
    }
  }
}

Machine::Machine(const std::map<TaskTarget, ProcessorRange>& ranges)
  : Machine{private_tag{}, ranges}
{
}

Machine::Machine(std::map<TaskTarget, ProcessorRange>&& ranges)
  : Machine{private_tag{}, std::move(ranges)}
{
}

const ProcessorRange& Machine::processor_range() const { return processor_range(preferred_target); }

namespace {

constexpr ProcessorRange EMPTY_RANGE{};
// clang-format likes to gobble this space unless we put something here...
}  // namespace

const ProcessorRange& Machine::processor_range(TaskTarget target) const
{
  auto finder = processor_ranges.find(target);
  if (finder == processor_ranges.end()) {
    return EMPTY_RANGE;
  }
  return finder->second;
}

std::vector<TaskTarget> Machine::valid_targets() const
{
  std::vector<TaskTarget> result;

  result.reserve(processor_ranges.size());
  for (auto& [target, range] : processor_ranges) {
    if (range.empty()) {
      continue;
    }
    result.push_back(target);
  }
  return result;
}

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  std::vector<TaskTarget> result;
  for (auto& [target, _] : processor_ranges) {
    if (to_exclude.find(target) == to_exclude.end()) {
      result.push_back(target);
    }
  }
  return result;
}

uint32_t Machine::count() const { return count(preferred_target); }

uint32_t Machine::count(TaskTarget target) const { return processor_range(target).count(); }

std::string Machine::to_string() const
{
  std::stringstream ss;
  ss << "Machine(preferred_target: " << preferred_target;
  for (auto& [kind, range] : processor_ranges) {
    ss << ", " << kind << ": " << range.to_string();
  }
  ss << ")";
  return std::move(ss).str();
}

void Machine::pack(legate::detail::BufferBuilder& buffer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(preferred_target));
  buffer.pack<uint32_t>(processor_ranges.size());
  for (auto& [target, processor_range] : processor_ranges) {
    buffer.pack<int32_t>(static_cast<int32_t>(target));
    buffer.pack<uint32_t>(processor_range.low);
    buffer.pack<uint32_t>(processor_range.high);
    buffer.pack<uint32_t>(processor_range.per_node_count);
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

Machine Machine::slice(uint32_t from, uint32_t to, TaskTarget target, bool keep_others) const
{
  if (keep_others) {
    std::map<TaskTarget, ProcessorRange> new_ranges{processor_ranges};

    new_ranges[target] = processor_range(target).slice(from, to);
    return Machine{std::move(new_ranges)};
  }
  return Machine{{{target, processor_range(target).slice(from, to)}}};
}

Machine Machine::slice(uint32_t from, uint32_t to, bool keep_others) const
{
  return slice(from, to, preferred_target, keep_others);
}

bool Machine::operator==(const Machine& other) const
{
  if (processor_ranges.size() < other.processor_ranges.size()) {
    return other.operator==(*this);
  }
  auto equal_ranges = [&](const auto& proc_range) {
    const auto& [target, range] = proc_range;

    if (range.empty()) {
      return true;
    }
    auto finder = other.processor_ranges.find(target);
    return !(finder == other.processor_ranges.end() || range != finder->second);
  };

  return std::all_of(processor_ranges.begin(), processor_ranges.end(), std::move(equal_ranges));
}

bool Machine::operator!=(const Machine& other) const { return !(*this == other); }

Machine Machine::operator&(const Machine& other) const
{
  std::map<TaskTarget, ProcessorRange> new_processor_ranges;
  for (const auto& [target, range] : processor_ranges) {
    auto finder = other.processor_ranges.find(target);
    if (finder != other.processor_ranges.end()) {
      new_processor_ranges[target] = finder->second & range;
    }
  }
  return Machine{std::move(new_processor_ranges)};
}

bool Machine::empty() const
{
  return std::all_of(
    processor_ranges.begin(), processor_ranges.end(), [](auto& rng) { return rng.second.empty(); });
}

std::ostream& operator<<(std::ostream& stream, const Machine& machine)
{
  stream << machine.to_string();
  return stream;
}

///////////////////////////////////////////
// legate::mapping::LocalProcessorRange
///////////////////////////////////////////

const Processor& LocalProcessorRange::operator[](uint32_t idx) const
{
  auto local_idx = idx - offset_;
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    static_assert(std::is_unsigned_v<decltype(local_idx)>,
                  "if local_idx becomes signed, also check local_idx >= 0 below!");
    assert(local_idx < procs_.size());
  }
  return procs_[local_idx];
}

std::string LocalProcessorRange::to_string() const
{
  std::stringstream ss;
  ss << "{offset: " << offset_ << ", total processor count: " << total_proc_count_
     << ", processors: ";
  for (auto&& proc : procs_) {
    ss << proc << ",";
  }
  ss << "}";
  return std::move(ss).str();
}

std::ostream& operator<<(std::ostream& stream, const LocalProcessorRange& range)
{
  stream << range.to_string();
  return stream;
}

///////////////////////////////////////////
// legate::mapping::LocalMachine
///////////////////////////////////////////
LocalMachine::LocalMachine()
  : node_id{static_cast<uint32_t>(Realm::Network::my_node_id)},
    total_nodes{static_cast<uint32_t>(Legion::Machine::get_machine().get_address_space_count())}
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
  assert(sysmem.count() > 0);
  system_memory_ = sysmem.first();

  if (!gpus_.empty()) {
    Legion::Machine::MemoryQuery zcmem{legion_machine};

    zcmem.local_address_space().only_kind(Legion::Memory::Z_COPY_MEM);
    assert(zcmem.count() > 0);
    zerocopy_memory_ = zcmem.first();
  }

  for (auto& gpu : gpus_) {
    Legion::Machine::MemoryQuery framebuffer{legion_machine};

    framebuffer.local_address_space().only_kind(Legion::Memory::GPU_FB_MEM).best_affinity_to(gpu);
    assert(framebuffer.count() > 0);
    frame_buffers_[gpu] = framebuffer.first();
  }

  for (auto& omp : omps_) {
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
}

const std::vector<Processor>& LocalMachine::procs(TaskTarget target) const
{
  switch (target) {
    case TaskTarget::GPU: return gpus_;
    case TaskTarget::OMP: return omps_;
    case TaskTarget::CPU: return cpus_;
    default: LEGATE_ABORT;
  }
  assert(false);
  return cpus_;
}

size_t LocalMachine::total_frame_buffer_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  const size_t per_node_size = frame_buffers_.size() * frame_buffers_.begin()->second.capacity();
  return per_node_size * total_nodes;
}

size_t LocalMachine::total_socket_memory_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  const size_t per_node_size =
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

  auto finder = machine.processor_ranges.find(target);
  if (machine.processor_ranges.end() == finder) {
    if (fallback_to_global) {
      return LocalProcessorRange{local_procs};
    }
    return {};
  }

  auto& global_range = finder->second;

  auto num_local_procs = local_procs.size();
  auto my_low          = num_local_procs * node_id;
  const ProcessorRange my_range{static_cast<uint32_t>(my_low),
                                static_cast<uint32_t>(my_low + num_local_procs),
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
    default: LEGATE_ABORT;
  }
  assert(false);
  return Legion::Memory::NO_MEMORY;
}

}  // namespace legate::mapping::detail
