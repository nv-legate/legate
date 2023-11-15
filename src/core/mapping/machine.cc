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

#include "core/mapping/machine.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace legate::mapping {

/////////////////////////////////////
// legate::mapping::ProcessorRange
/////////////////////////////////////

ProcessorRange ProcessorRange::slice(uint32_t from, uint32_t to) const
{
  auto new_low  = std::min(low + from, high);
  auto new_high = std::min(low + to, high);
  return {new_low, new_high, per_node_count};
}

NodeRange ProcessorRange::get_node_range() const
{
  if (empty()) {
    throw std::invalid_argument{"Illegal to get a node range of an empty processor range"};
  }
  return {low / per_node_count, (high + per_node_count - 1) / per_node_count};
}

std::string ProcessorRange::to_string() const
{
  std::stringstream ss;
  ss << "Proc([" << low << "," << high << "], " << per_node_count << " per node)";
  return std::move(ss).str();
}

ProcessorRange ProcessorRange::operator&(const ProcessorRange& other) const
{
  if (other.per_node_count != per_node_count) {
    throw std::invalid_argument{
      "Invalid to compute an intersection between processor ranges with different per-node counts"};
  }
  return {std::max(low, other.low), std::min(high, other.high), per_node_count};
}

bool ProcessorRange::operator<(const ProcessorRange& other) const noexcept
{
  if (low < other.low) {
    return true;
  }
  if (low > other.low) {
    return false;
  }
  if (high < other.high) {
    return true;
  }
  if (high > other.high) {
    return false;
  }
  return per_node_count < other.per_node_count;
}

std::ostream& operator<<(std::ostream& stream, const ProcessorRange& range)
{
  stream << range.to_string();
  return stream;
}

///////////////////////////////////////////
// legate::mapping::Machine
//////////////////////////////////////////

TaskTarget Machine::preferred_target() const { return impl()->preferred_target; }

ProcessorRange Machine::processor_range() const { return processor_range(preferred_target()); }

ProcessorRange Machine::processor_range(TaskTarget target) const
{
  return impl()->processor_range(target);
}

std::vector<TaskTarget> Machine::valid_targets() const { return impl()->valid_targets(); }

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  return impl()->valid_targets_except(to_exclude);
}

uint32_t Machine::count() const { return count(preferred_target()); }

uint32_t Machine::count(TaskTarget target) const { return impl()->count(target); }

std::string Machine::to_string() const { return impl()->to_string(); }

Machine Machine::only(TaskTarget target) const { return only(std::vector<TaskTarget>{target}); }

Machine Machine::only(const std::vector<TaskTarget>& targets) const
{
  return Machine{impl()->only(targets)};
}

Machine Machine::slice(uint32_t from, uint32_t to, TaskTarget target, bool keep_others) const
{
  return Machine{impl()->slice(from, to, target, keep_others)};
}

Machine Machine::slice(uint32_t from, uint32_t to, bool keep_others) const
{
  return slice(from, to, preferred_target(), keep_others);
}

Machine Machine::operator[](TaskTarget target) const { return only(target); }

Machine Machine::operator[](const std::vector<TaskTarget>& targets) const { return only(targets); }

bool Machine::operator==(const Machine& other) const { return *impl() == *other.impl(); }

bool Machine::operator!=(const Machine& other) const { return !(*impl() == *other.impl()); }

Machine Machine::operator&(const Machine& other) const { return Machine{*impl() & *other.impl()}; }

bool Machine::empty() const { return impl()->empty(); }

Machine::Machine(std::shared_ptr<detail::Machine> impl) : impl_{std::move(impl)} {}

Machine::Machine(detail::Machine impl) : Machine{std::make_shared<detail::Machine>(std::move(impl))}
{
}

Machine::Machine(std::map<TaskTarget, ProcessorRange> ranges)
  : Machine{detail::Machine{std::move(ranges)}}
{
}

std::ostream& operator<<(std::ostream& stream, const Machine& machine)
{
  stream << machine.impl()->to_string();
  return stream;
}

}  // namespace legate::mapping
