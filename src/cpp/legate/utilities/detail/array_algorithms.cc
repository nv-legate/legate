/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/array_algorithms.h>

#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <stdexcept>
#include <vector>

namespace legate::detail {

void assert_valid_mapping(std::size_t container_size, Span<const std::int32_t> mapping)
{
  if (mapping.size() != container_size) {
    throw TracedException<std::out_of_range>{
      fmt::format("mapping size {} != container size {}", mapping.size(), container_size)};
  }

  // Early out here because we use front() and back() below
  if (mapping.empty()) {
    return;
  }

  auto sorted_mapping = std::vector<std::int32_t>{mapping.begin(), mapping.end()};

  std::sort(sorted_mapping.begin(), sorted_mapping.end());
  // Check that elements are in range. The copy is sorted, so it suffices to check the
  // bounds. If either is out of range, then at least one element of the mapping is out of
  // range.
  if (sorted_mapping.front() < 0) {
    throw TracedException<std::out_of_range>{
      fmt::format("mapping {} contains negative elements", mapping)};
  }
  if (static_cast<std::size_t>(sorted_mapping.back()) >= container_size) {
    throw TracedException<std::out_of_range>{fmt::format(
      "mapping {} contains elements outside of container size {}", mapping, container_size)};
  }

  // Check that elements are unique
  if (const auto it = std::adjacent_find(sorted_mapping.begin(), sorted_mapping.end());
      it != sorted_mapping.end()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid mapping: contains duplicate element(s) {} ({})", *it, mapping)};
  }
}

void assert_in_range(std::size_t container_size, std::int64_t pos)
{
  if ((pos < 0) || (static_cast<std::size_t>(pos) >= container_size)) {
    throw TracedException<std::out_of_range>{
      fmt::format("Index {} out of range [0, {})", pos, container_size)};
  }
}

}  // namespace legate::detail
