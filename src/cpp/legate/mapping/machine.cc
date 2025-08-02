/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/machine.h>

#include <legate/mapping/detail/machine.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace legate::mapping {

std::size_t NodeRange::hash() const noexcept { return hash_all(low, high); }

/////////////////////////////////////
// legate::mapping::ProcessorRange
/////////////////////////////////////

/*static*/ void ProcessorRange::throw_illegal_empty_node_range_()
{
  throw legate::detail::TracedException<std::invalid_argument>{
    "Illegal to get a node range of an empty processor range"};
}

/*static*/ void ProcessorRange::throw_illegal_invalid_intersection_()
{
  throw legate::detail::TracedException<std::invalid_argument>{
    "Invalid to compute an intersection between processor ranges with different per-node counts"};
}

std::string ProcessorRange::to_string() const
{
  return fmt::format("Proc([{},{}], {} per node)", low, high, per_node_count);
}

std::size_t ProcessorRange::hash() const noexcept { return hash_all(low, high, per_node_count); }

std::ostream& operator<<(std::ostream& stream, const ProcessorRange& range)
{
  stream << range.to_string();
  return stream;
}

///////////////////////////////////////////
// legate::mapping::Machine
//////////////////////////////////////////

TaskTarget Machine::preferred_target() const { return impl()->preferred_target(); }

ProcessorRange Machine::processor_range() const { return processor_range(preferred_target()); }

ProcessorRange Machine::processor_range(TaskTarget target) const
{
  return impl()->processor_range(target);
}

Span<const TaskTarget> Machine::valid_targets() const { return impl()->valid_targets(); }

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  auto&& ret = impl()->valid_targets_except(to_exclude);

  return {ret.begin(), ret.end()};
}

std::uint32_t Machine::count() const { return count(preferred_target()); }

std::uint32_t Machine::count(TaskTarget target) const { return impl()->count(target); }

std::string Machine::to_string() const { return impl()->to_string(); }

Machine Machine::only(TaskTarget target) const { return Machine{impl()->only(target)}; }

Machine Machine::only(Span<const TaskTarget> targets) const
{
  return Machine{impl()->only(targets)};
}

Machine Machine::slice(std::uint32_t from,
                       std::uint32_t to,
                       TaskTarget target,
                       bool keep_others) const
{
  return Machine{impl()->slice(from, to, target, keep_others)};
}

Machine Machine::slice(std::uint32_t from, std::uint32_t to, bool keep_others) const
{
  return slice(from, to, preferred_target(), keep_others);
}

Machine Machine::operator[](TaskTarget target) const { return Machine{(*impl())[target]}; }

Machine Machine::operator[](Span<const TaskTarget> targets) const
{
  return Machine{(*impl())[targets]};
}

bool Machine::operator==(const Machine& other) const { return *impl() == *other.impl(); }

bool Machine::operator!=(const Machine& other) const { return !(*impl() == *other.impl()); }

Machine Machine::operator&(const Machine& other) const { return Machine{*impl() & *other.impl()}; }

bool Machine::empty() const { return impl()->empty(); }

Machine::Machine(InternalSharedPtr<detail::Machine> impl) : impl_{std::move(impl)} {}

Machine::Machine(detail::Machine impl)
  : Machine{make_internal_shared<detail::Machine>(std::move(impl))}
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
