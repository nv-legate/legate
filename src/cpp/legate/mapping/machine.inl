/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/machine.h>
#include <legate/utilities/hash.h>

#include <algorithm>

namespace legate::mapping {

/////////////////////////////////////
// legate::mapping::NodeRange
/////////////////////////////////////

constexpr bool NodeRange::operator<(const NodeRange& other) const noexcept
{
  return low < other.low || (low == other.low && high < other.high);
}

constexpr bool NodeRange::operator==(const NodeRange& other) const noexcept
{
  return low == other.low && high == other.high;
}

constexpr bool NodeRange::operator!=(const NodeRange& other) const noexcept
{
  return !(other == *this);
}

/////////////////////////////////////
// legate::mapping::ProcessorRange
/////////////////////////////////////

constexpr std::uint32_t ProcessorRange::count() const noexcept { return high - low; }

constexpr bool ProcessorRange::empty() const noexcept { return high <= low; }

constexpr ProcessorRange ProcessorRange::slice(std::uint32_t from, std::uint32_t to) const
{
  const auto new_low  = std::min(low + from, high);
  const auto new_high = std::min(low + to, high);
  return {new_low, new_high, per_node_count};
}

constexpr NodeRange ProcessorRange::get_node_range() const
{
  if (empty()) {
    throw_illegal_empty_node_range_();
  }
  return {low / per_node_count, (high + per_node_count - 1) / per_node_count};
}

constexpr ProcessorRange::ProcessorRange(std::uint32_t low_id,
                                         std::uint32_t high_id,
                                         std::uint32_t per_node_proc_count) noexcept
  : low{low_id < high_id ? low_id : 0},
    high{low_id < high_id ? high_id : 0},
    per_node_count{std::max(std::uint32_t{1}, per_node_proc_count)}
{
}

constexpr ProcessorRange ProcessorRange::operator&(const ProcessorRange& other) const
{
  if (other.per_node_count != per_node_count) {
    throw_illegal_invalid_intersection_();
  }
  return {std::max(low, other.low), std::min(high, other.high), per_node_count};
}

constexpr bool ProcessorRange::operator==(const ProcessorRange& other) const noexcept
{
  return other.low == low && other.high == high && other.per_node_count == per_node_count;
}

constexpr bool ProcessorRange::operator!=(const ProcessorRange& other) const noexcept
{
  return !(other == *this);
}

constexpr bool ProcessorRange::operator<(const ProcessorRange& other) const noexcept
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

///////////////////////////////////////////
// legate::mapping::Machine
//////////////////////////////////////////

inline const SharedPtr<detail::Machine>& Machine::impl() const { return impl_; }

}  // namespace legate::mapping
