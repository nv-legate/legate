/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/partition_placement_info.h>

#include <legate/data/detail/partition_placement.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/machine.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

namespace legate::detail {

std::optional<std::reference_wrapper<const PartitionPlacement>>
PartitionPlacementInfo::get_placement_for_color(Span<const std::uint64_t> color) const
{
  auto it = std::find_if(
    placements().begin(), placements().end(), [&color](const PartitionPlacement& mapping) {
      return std::equal(mapping.partition_color().begin(),
                        mapping.partition_color().end(),
                        color.begin(),
                        color.end());
    });

  if (it != placements().end()) {
    return std::ref(*it);
  }
  return std::nullopt;
}

std::string PartitionPlacementInfo::to_string() const
{
  // Calculate initial column widths based on column names
  std::size_t color_width  = std::string{"Partition Color"}.length();
  std::size_t node_width   = std::string{"Node"}.length();
  std::size_t memory_width = std::string{"Memory"}.length();

  // Collect all formatted strings and calculate max widths
  std::vector<std::string> color_strs{};
  std::vector<std::string> node_strs{};

  for (const auto& mapping : placements_) {
    // Format partition color using fmt::join
    auto color_str = fmt::format("{}", mapping.partition_color());
    color_strs.push_back(color_str);
    color_width = std::max(color_width, color_str.length());

    auto node_str = fmt::format("{}", mapping.node_id());
    node_strs.push_back(node_str);
    node_width = std::max(node_width, node_str.length());

    // Calculate width for memory type using formatter
    auto memory_width_temp = fmt::formatted_size("{}", mapping.memory_type());
    memory_width           = std::max(memory_width, memory_width_temp);
  }

  std::string result;

  // Create top border
  fmt::format_to(std::back_inserter(result),
                 "+-{:-<{}}-+-{:-<{}}-+-{:-<{}}-+\n",
                 "",
                 color_width,
                 "",
                 node_width,
                 "",
                 memory_width);

  // Create header row
  fmt::format_to(std::back_inserter(result),
                 "| {:<{}} | {:<{}} | {:<{}} |\n",
                 "Partition Color",
                 color_width,
                 "Node",
                 node_width,
                 "Memory",
                 memory_width);

  // Create separator
  fmt::format_to(std::back_inserter(result),
                 "+-{:-<{}}-+-{:-<{}}-+-{:-<{}}-+\n",
                 "",
                 color_width,
                 "",
                 node_width,
                 "",
                 memory_width);

  // Create data rows
  for (std::size_t i = 0; i < placements_.size(); ++i) {
    fmt::format_to(std::back_inserter(result),
                   "| {:<{}} | {:<{}} | {:<{}} |\n",
                   color_strs[i],
                   color_width,
                   node_strs[i],
                   node_width,
                   placements_[i].memory_type(),
                   memory_width);
  }

  if (!placements_.empty()) {
    // Create bottom border
    fmt::format_to(std::back_inserter(result),
                   "+-{:-<{}}-+-{:-<{}}-+-{:-<{}}-+\n",
                   "",
                   color_width,
                   "",
                   node_width,
                   "",
                   memory_width);
  }

  return result;
}

}  // namespace legate::detail
