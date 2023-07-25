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

#include "core/mapping/machine.h"
#include "core/mapping/detail/machine.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::mapping {

/////////////////////////////////////
// legate::mapping::NodeRange
/////////////////////////////////////

bool NodeRange::operator<(const NodeRange& other) const
{
  return low < other.low || (low == other.low && high < other.high);
}

bool NodeRange::operator==(const NodeRange& other) const
{
  return low == other.low && high == other.high;
}

bool NodeRange::operator!=(const NodeRange& other) const { return !operator==(other); }

/////////////////////////////////////
// legate::mapping::ProcessorRange
/////////////////////////////////////

uint32_t ProcessorRange::count() const { return high - low; }

bool ProcessorRange::empty() const { return high <= low; }

ProcessorRange ProcessorRange::slice(uint32_t from, uint32_t to) const
{
  uint32_t new_low  = std::min<uint32_t>(low + from, high);
  uint32_t new_high = std::min<uint32_t>(low + to, high);
  return ProcessorRange(new_low, new_high, per_node_count);
}

NodeRange ProcessorRange::get_node_range() const
{
  if (empty()) {
    throw std::invalid_argument("Illegal to get a node range of an empty processor range");
  }
  return NodeRange{low / per_node_count, (high + per_node_count - 1) / per_node_count};
}

std::string ProcessorRange::to_string() const
{
  std::stringstream ss;
  ss << "Proc([" << low << "," << high << "], " << per_node_count << " per node)";
  return ss.str();
}

ProcessorRange::ProcessorRange() {}

ProcessorRange::ProcessorRange(uint32_t _low, uint32_t _high, uint32_t _per_node_count)
  : low(_low < _high ? _low : 0),
    high(_low < _high ? _high : 0),
    per_node_count(std::max<uint32_t>(1, _per_node_count))
{
}

ProcessorRange ProcessorRange::operator&(const ProcessorRange& other) const
{
  if (other.per_node_count != per_node_count) {
    throw std::invalid_argument(
      "Invalid to compute an intersection between processor ranges with different per-node counts");
  }
  return ProcessorRange(std::max(low, other.low), std::min(high, other.high), per_node_count);
}

bool ProcessorRange::operator==(const ProcessorRange& other) const
{
  return other.low == low && other.high == high && other.per_node_count == per_node_count;
}

bool ProcessorRange::operator!=(const ProcessorRange& other) const { return !operator==(other); }

bool ProcessorRange::operator<(const ProcessorRange& other) const
{
  if (low < other.low)
    return true;
  else if (low > other.low)
    return false;
  if (high < other.high)
    return true;
  else if (high > other.high)
    return false;
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

TaskTarget Machine::preferred_target() const { return impl_->preferred_target; }

ProcessorRange Machine::processor_range() const { return processor_range(impl_->preferred_target); }
ProcessorRange Machine::processor_range(TaskTarget target) const
{
  return impl_->processor_range(target);
}

std::vector<TaskTarget> Machine::valid_targets() const { return impl_->valid_targets(); }

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  return impl_->valid_targets_except(to_exclude);
}

uint32_t Machine::count() const { return count(impl_->preferred_target); }

uint32_t Machine::count(TaskTarget target) const { return impl_->count(target); }

std::string Machine::to_string() const { return impl_->to_string(); }

Machine Machine::only(TaskTarget target) const { return only(std::vector({target})); }

Machine Machine::only(const std::vector<TaskTarget>& targets) const
{
  return Machine(new detail::Machine(impl_->only(targets)));
}

Machine Machine::slice(uint32_t from, uint32_t to, TaskTarget target, bool keep_others) const
{
  return Machine(new detail::Machine(impl_->slice(from, to, target, keep_others)));
}

Machine Machine::slice(uint32_t from, uint32_t to, bool keep_others) const
{
  return slice(from, to, impl_->preferred_target, keep_others);
}

Machine Machine::operator[](TaskTarget target) const { return only(target); }

Machine Machine::operator[](const std::vector<TaskTarget>& targets) const { return only(targets); }

bool Machine::operator==(const Machine& other) const { return *impl_ == *other.impl_; }

bool Machine::operator!=(const Machine& other) const { return !(*impl_ == *other.impl_); }

Machine Machine::operator&(const Machine& other) const
{
  auto result = *impl_ & *other.impl_;
  return Machine(new detail::Machine(std::move(result)));
}

bool Machine::empty() const { return impl_->empty(); }

Machine::Machine(detail::Machine* impl) : impl_(impl) {}

Machine::Machine(const detail::Machine& impl) : impl_(new detail::Machine(impl)) {}

Machine::Machine(const Machine& other) : impl_(new detail::Machine(*other.impl_)) {}

Machine& Machine::operator=(const Machine& other)
{
  impl_ = new detail::Machine(*other.impl_);
  return *this;
}

Machine::Machine(Machine&& other) : impl_(other.impl_) { other.impl_ = nullptr; }

Machine& Machine::operator=(Machine&& other)
{
  impl_       = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

Machine::~Machine() { delete impl_; }

std::ostream& operator<<(std::ostream& stream, const Machine& machine)
{
  stream << machine.impl()->to_string();
  return stream;
}

}  // namespace legate::mapping
