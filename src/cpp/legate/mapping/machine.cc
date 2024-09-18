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

#include "legate/mapping/machine.h"

#include "legate/mapping/detail/machine.h"

#include <fmt/format.h>
#include <iostream>
#include <utility>

namespace legate::mapping {

std::size_t NodeRange::hash() const noexcept { return hash_all(low, high); }

/////////////////////////////////////
// legate::mapping::ProcessorRange
/////////////////////////////////////

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

const std::vector<TaskTarget>& Machine::valid_targets() const { return impl()->valid_targets(); }

std::vector<TaskTarget> Machine::valid_targets_except(const std::set<TaskTarget>& to_exclude) const
{
  return impl()->valid_targets_except(to_exclude);
}

std::uint32_t Machine::count() const { return count(preferred_target()); }

std::uint32_t Machine::count(TaskTarget target) const { return impl()->count(target); }

std::string Machine::to_string() const { return impl()->to_string(); }

Machine Machine::only(TaskTarget target) const { return only(std::vector<TaskTarget>{target}); }

Machine Machine::only(const std::vector<TaskTarget>& targets) const
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

Machine Machine::operator[](TaskTarget target) const { return only(target); }

Machine Machine::operator[](const std::vector<TaskTarget>& targets) const { return only(targets); }

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
