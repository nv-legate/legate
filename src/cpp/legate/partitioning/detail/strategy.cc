/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/strategy.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/formatters.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace legate::detail {

void Strategy::insert(const Variable& partition_symbol, InternalSharedPtr<Partition> partition)
{
  LEGATE_ASSERT(!has_assignment(partition_symbol));
  assignments_.insert({partition_symbol, std::move(partition)});
}

void Strategy::insert(const Variable& partition_symbol,
                      Legion::FieldSpace field_space,
                      Legion::FieldID field_id)
{
  LEGATE_ASSERT(fields_for_unbound_stores_.find(partition_symbol) ==
                fields_for_unbound_stores_.end());
  fields_for_unbound_stores_.insert({partition_symbol, {field_space, field_id}});
}

bool Strategy::has_assignment(const Variable& partition_symbol) const
{
  return assignments_.find(partition_symbol) != assignments_.end();
}

const InternalSharedPtr<Partition>& Strategy::operator[](const Variable& partition_symbol) const
{
  const auto it = assignments_.find(partition_symbol);

  LEGATE_ASSERT(it != assignments_.end());
  return it->second;
}

const std::pair<Legion::FieldSpace, Legion::FieldID>& Strategy::find_field_for_unbound_store(
  const Variable& partition_symbol) const
{
  const auto finder = fields_for_unbound_stores_.find(partition_symbol);

  LEGATE_ASSERT(finder != fields_for_unbound_stores_.end());
  return finder->second;
}

void Strategy::dump() const
{
  if (!log_legate_partitioner().want_debug()) {
    return;
  }
  log_legate_partitioner().debug() << "===== Solution =====";
  if (!launch_domain_.is_valid()) {
    log_legate_partitioner().debug()
      << operation_->to_string(true /*show_provenance*/) << ": (sequential)";
  } else {
    log_legate_partitioner().debug()
      << operation_->to_string(true /*show_provenance*/) << ": " << launch_domain_;
  }
  for (const auto& [symbol, part] : assignments_) {
    log_legate_partitioner().debug() << symbol.to_string() << ": " << part->to_string();
  }
  for (const auto& [symbol, field] : fields_for_unbound_stores_) {
    const auto& [field_space, field_id] = field;

    log_legate_partitioner().debug()
      << symbol.to_string() << ": (" << field_space << "," << field_id << ")";
  }
  log_legate_partitioner().debug() << "====================";
}

void Strategy::record_key_partition(PrivateKey, const Variable& partition_symbol)
{
  if (!key_partition_.has_value()) {
    key_partition_ = partition_symbol;
  }
}

Legion::ProjectionID Strategy::find_store_projection(const Variable& partition_symbol) const
{
  auto finder = projection_ids_.find(partition_symbol);

  if (finder == projection_ids_.end()) {
    return Legion::ProjectionID{0};
  }

  return finder->second;
}

Legion::ProjectionID Strategy::find_key_store_projection() const
{
  if (!key_partition_.has_value()) {
    // If no key partition exists, use the identity function (of ID 0)
    return 0;
  }

  return find_store_projection(*key_partition_);
}

}  // namespace legate::detail
