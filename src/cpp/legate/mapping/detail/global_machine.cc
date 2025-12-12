/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/global_machine.h>

#include <legate/utilities/detail/formatters.h>

#include <realm/network.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <unistd.h>

namespace legate::mapping::detail {

GlobalMachine::GlobalMachine()
  : total_nodes_{
      static_cast<std::uint32_t>(Legion::Machine::get_machine().get_address_space_count())}
{
  auto legion_machine = Legion::Machine::get_machine();
  const Legion::Machine::ProcessorQuery procs{legion_machine};

  for (auto proc : procs) {
    switch (proc.kind()) {
      case Processor::LOC_PROC: {
        global_cpus_.push_back(proc);
        continue;
      }
      case Processor::TOC_PROC: {
        global_gpus_.push_back(proc);
        continue;
      }
      case Processor::OMP_PROC: {
        global_omps_.push_back(proc);
        continue;
      }
      case Processor::NO_KIND: [[fallthrough]];
      case Processor::UTIL_PROC: [[fallthrough]];
      case Processor::IO_PROC: [[fallthrough]];
      case Processor::PROC_GROUP: [[fallthrough]];
      case Processor::PROC_SET: [[fallthrough]];
      case Processor::PY_PROC: continue;
    }
  }
}

legate::Span<const Processor> GlobalMachine::procs(TaskTarget target) const
{
  switch (target) {
    case TaskTarget::GPU: return global_gpus_;
    case TaskTarget::OMP: return global_omps_;
    case TaskTarget::CPU: return global_cpus_;
  }
  LEGATE_ABORT(fmt::format("GlobalMachine::procs not implemented for target: {}", target));
}

ProcessorSpan GlobalMachine::slice(const Machine& machine) const
{
  const auto preferred_target = machine.preferred_target();
  const auto& global_procs    = procs(preferred_target);

  // Find processors of target in machine argument
  auto it = machine.processor_ranges().find(preferred_target);

  if (machine.processor_ranges().end() == it) {
    return {};
  }

  auto& global_range = it->second;

  auto num_global_procs = global_procs.size();
  const ProcessorRange my_range{
    /*low_id=*/0, static_cast<std::uint32_t>(num_global_procs), global_range.per_node_count};

  // Check intersection between global machine parameter and global list of processors
  auto slice = global_range & my_range;

  if (slice.empty()) {
    return {};
  }

  const Span<const Processor> procs{global_procs.data() + slice.low, slice.count()};

  return {slice.low, global_range.count(), procs};
}

ProcessorSpan GlobalMachine::slice_with_fallback(const Machine& machine) const
{
  const auto preferred_target = machine.preferred_target();
  const auto& global_procs    = procs(preferred_target);

  // Find processors of target in machine argument (fallback if none in machine)
  auto it = machine.processor_ranges().find(preferred_target);

  if (machine.processor_ranges().end() == it) {
    return ProcessorSpan{global_procs};
  }

  auto& global_range = it->second;

  auto num_global_procs = global_procs.size();
  const ProcessorRange my_range{
    /*low_id=*/0, static_cast<std::uint32_t>(num_global_procs), global_range.per_node_count};

  // Check intersection between global machine parameter and global list of processors
  auto slice = global_range & my_range;

  if (slice.empty()) {
    return ProcessorSpan{global_procs};
  }

  const Span<const Processor> procs{global_procs.data() + slice.low, slice.count()};

  return {slice.low, global_range.count(), procs};
}

}  // namespace legate::mapping::detail
