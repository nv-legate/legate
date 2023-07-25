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

#include "core/mapping/detail/machine.h"
#include "core/utilities/detail/buffer_builder.h"
#include "realm/network.h"

namespace legate::mapping::detail {

///////////////////////////////////////////
// legate::mapping::detail::Machine
//////////////////////////////////////////

Machine::Machine(const std::map<TaskTarget, ProcessorRange>& ranges) : processor_ranges(ranges)
{
  for (auto& [target, processor_range] : processor_ranges)
    if (!processor_range.empty()) {
      preferred_target = target;
      return;
    }
}

Machine::Machine(std::map<TaskTarget, ProcessorRange>&& ranges)
  : processor_ranges(std::move(ranges))
{
  for (auto& [target, processor_range] : processor_ranges)
    if (!processor_range.empty()) {
      preferred_target = target;
      return;
    }
}

const ProcessorRange& Machine::processor_range() const { return processor_range(preferred_target); }
static const ProcessorRange EMPTY_RANGE{};

const ProcessorRange& Machine::processor_range(TaskTarget target) const
{
  auto finder = processor_ranges.find(target);
  if (finder == processor_ranges.end()) { return EMPTY_RANGE; }
  return finder->second;
}

std::vector<TaskTarget> Machine::valid_targets() const
{
  std::vector<TaskTarget> result;
  for (auto& [target, _] : processor_ranges) result.push_back(target);
  return result;
}

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  std::vector<TaskTarget> result;
  for (auto& [target, _] : processor_ranges)
    if (to_exclude.find(target) == to_exclude.end()) result.push_back(target);
  return result;
}

uint32_t Machine::count() const { return count(preferred_target); }

uint32_t Machine::count(TaskTarget target) const { return processor_range(target).count(); }

std::string Machine::to_string() const
{
  std::stringstream ss;
  ss << "Machine(preferred_target: " << preferred_target;
  for (auto& [kind, range] : processor_ranges) ss << ", " << kind << ": " << range.to_string();
  ss << ")";
  return ss.str();
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

Machine Machine::only(TaskTarget target) const { return only(std::vector({target})); }

Machine Machine::only(const std::vector<TaskTarget>& targets) const
{
  std::map<TaskTarget, ProcessorRange> new_processor_ranges;
  for (auto t : targets) new_processor_ranges.insert({t, processor_range(t)});

  return Machine(new_processor_ranges);
}

Machine Machine::slice(uint32_t from, uint32_t to, TaskTarget target, bool keep_others) const
{
  if (keep_others) {
    std::map<TaskTarget, ProcessorRange> new_ranges(processor_ranges);
    new_ranges[target] = processor_range(target).slice(from, to);
    return Machine(std::move(new_ranges));
  } else
    return Machine({{target, processor_range(target).slice(from, to)}});
}

Machine Machine::slice(uint32_t from, uint32_t to, bool keep_others) const
{
  return slice(from, to, preferred_target, keep_others);
}

Machine Machine::operator[](TaskTarget target) const { return only(target); }

Machine Machine::operator[](const std::vector<TaskTarget>& targets) const { return only(targets); }

bool Machine::operator==(const Machine& other) const
{
  if (preferred_target != other.preferred_target) return false;
  if (processor_ranges.size() != other.processor_ranges.size()) return false;
  for (auto const& r : processor_ranges) {
    auto finder = other.processor_ranges.find(r.first);
    if (finder == other.processor_ranges.end() || r.second != finder->second) return false;
  }
  return true;
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
  return Machine(new_processor_ranges);
}

bool Machine::empty() const
{
  for (const auto& r : processor_ranges)
    if (!r.second.empty()) return false;
  return true;
}

std::ostream& operator<<(std::ostream& stream, const Machine& machine)
{
  stream << machine.to_string();
  return stream;
}

///////////////////////////////////////////
// legate::mapping::LocalProcessorRange
///////////////////////////////////////////

LocalProcessorRange::LocalProcessorRange() : offset_(0), total_proc_count_(0), procs_() {}

LocalProcessorRange::LocalProcessorRange(const std::vector<Processor>& procs)
  : offset_(0), total_proc_count_(procs.size()), procs_(procs.data(), procs.size())
{
}

LocalProcessorRange::LocalProcessorRange(uint32_t offset,
                                         uint32_t total_proc_count,
                                         const Processor* local_procs,
                                         size_t num_local_procs)
  : offset_(offset), total_proc_count_(total_proc_count), procs_(local_procs, num_local_procs)
{
}

const Processor& LocalProcessorRange::operator[](uint32_t idx) const
{
  auto local_idx = (idx % total_proc_count_) - offset_;
#ifdef DEBUG_LEGATE
  assert(local_idx < procs_.size());
#endif
  return procs_[local_idx];
}

std::string LocalProcessorRange::to_string() const
{
  std::stringstream ss;
  ss << "{offset: " << offset_ << ", total processor count: " << total_proc_count_
     << ", processors: ";
  for (uint32_t idx = 0; idx < procs_.size(); ++idx) ss << procs_[idx] << ",";
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
  : node_id(Realm::Network::my_node_id),
    total_nodes(Legion::Machine::get_machine().get_address_space_count())
{
  auto legion_machine = Legion::Machine::get_machine();
  Legion::Machine::ProcessorQuery procs(legion_machine);
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
  Legion::Machine::MemoryQuery sysmem(legion_machine);
  sysmem.local_address_space().only_kind(Legion::Memory::SYSTEM_MEM);
  assert(sysmem.count() > 0);
  system_memory_ = sysmem.first();

  if (!gpus_.empty()) {
    Legion::Machine::MemoryQuery zcmem(legion_machine);
    zcmem.local_address_space().only_kind(Legion::Memory::Z_COPY_MEM);
    assert(zcmem.count() > 0);
    zerocopy_memory_ = zcmem.first();
  }
  for (auto& gpu : gpus_) {
    Legion::Machine::MemoryQuery framebuffer(legion_machine);
    framebuffer.local_address_space().only_kind(Legion::Memory::GPU_FB_MEM).best_affinity_to(gpu);
    assert(framebuffer.count() > 0);
    frame_buffers_[gpu] = framebuffer.first();
  }
  for (auto& omp : omps_) {
    Legion::Machine::MemoryQuery sockmem(legion_machine);
    sockmem.local_address_space().only_kind(Legion::Memory::SOCKET_MEM).best_affinity_to(omp);
    // If we have socket memories then use them
    if (sockmem.count() > 0) socket_memories_[omp] = sockmem.first();
    // Otherwise we just use the local system memory
    else
      socket_memories_[omp] = system_memory_;
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
  size_t per_node_size = frame_buffers_.size() * frame_buffers_.begin()->second.capacity();
  return per_node_size * total_nodes;
}

size_t LocalMachine::total_socket_memory_size() const
{
  // We assume that all memories of the same kind are symmetric in size
  size_t per_node_size = socket_memories_.size() * socket_memories_.begin()->second.capacity();
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
    if (fallback_to_global)
      return LocalProcessorRange(local_procs);
    else
      return LocalProcessorRange();
  }

  auto& global_range = finder->second;

  uint32_t num_local_procs = local_procs.size();
  uint32_t my_low          = num_local_procs * node_id;
  ProcessorRange my_range(my_low, my_low + num_local_procs, global_range.per_node_count);

  auto slice = global_range & my_range;
  if (slice.empty()) {
    if (fallback_to_global)
      return LocalProcessorRange(local_procs);
    else
      return LocalProcessorRange();
  }

  return LocalProcessorRange(slice.low - global_range.low,
                             global_range.count(),
                             local_procs.data() + (slice.low - my_low),
                             slice.count());
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
