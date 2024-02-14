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

#include "core/mapping/machine.h"

#include <algorithm>

namespace legate::mapping {

/////////////////////////////////////
// legate::mapping::NodeRange
/////////////////////////////////////

inline bool NodeRange::operator<(const NodeRange& other) const noexcept
{
  return low < other.low || (low == other.low && high < other.high);
}

inline bool NodeRange::operator==(const NodeRange& other) const noexcept
{
  return low == other.low && high == other.high;
}

inline bool NodeRange::operator!=(const NodeRange& other) const noexcept
{
  return !(other == *this);
}

/////////////////////////////////////
// legate::mapping::ProcessorRange
/////////////////////////////////////

inline std::uint32_t ProcessorRange::count() const noexcept { return high - low; }

inline bool ProcessorRange::empty() const noexcept { return high <= low; }

inline bool ProcessorRange::operator==(const ProcessorRange& other) const noexcept
{
  return other.low == low && other.high == high && other.per_node_count == per_node_count;
}

inline bool ProcessorRange::operator!=(const ProcessorRange& other) const noexcept
{
  return !(other == *this);
}

inline ProcessorRange::ProcessorRange(std::uint32_t low_id,
                                      std::uint32_t high_id,
                                      std::uint32_t per_node_proc_count) noexcept
  : low{low_id < high_id ? low_id : 0},
    high{low_id < high_id ? high_id : 0},
    per_node_count{std::max(uint32_t{1}, per_node_proc_count)}
{
}

///////////////////////////////////////////
// legate::mapping::Machine
//////////////////////////////////////////

inline const SharedPtr<detail::Machine>& Machine::impl() const { return impl_; }

}  // namespace legate::mapping
