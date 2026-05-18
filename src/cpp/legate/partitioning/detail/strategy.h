/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/constraint.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <optional>
#include <unordered_map>
#include <utility>

namespace legate::detail {

class Operation;
class Partition;
class ConstraintSolver;

class Strategy {
 public:
  explicit Strategy(const Operation* operation);

  // NOTE(amberhassaan): disabled only because copy is expensive.
  Strategy(const Strategy&)            = delete;
  Strategy& operator=(const Strategy&) = delete;

  Strategy(Strategy&&) noexcept            = default;
  Strategy& operator=(Strategy&&) noexcept = default;

  [[nodiscard]] bool parallel() const;
  [[nodiscard]] const Domain& launch_domain() const;
  void set_launch_domain(const Domain& launch_domain);

  void insert(const Variable& partition_symbol, InternalSharedPtr<Partition> partition);
  void insert(const Variable& partition_symbol,
              Legion::FieldSpace field_space,
              Legion::FieldID field_id);
  void insert_color_space(const Variable& partition_symbol, Legion::IndexSpace color_space);
  [[nodiscard]] bool has_assignment(const Variable& partition_symbol) const;
  [[nodiscard]] const InternalSharedPtr<Partition>& operator[](
    const Variable& partition_symbol) const;
  [[nodiscard]] const std::pair<Legion::FieldSpace, Legion::FieldID>& find_field_for_unbound_store(
    const Variable& partition_symbol) const;

  void dump() const;

  class PrivateKey {
    friend class Fill;
    friend class ManualTask;
    friend class Strategy;
    friend class Partitioner;
  };

  void record_key_partition(PrivateKey, const Variable& partition_symbol);
  void insert_store_projection(PrivateKey,
                               const Variable& partition_symbol,
                               Legion::ProjectionID projection_id);
  [[nodiscard]] Legion::ProjectionID find_store_projection(const Variable& partition_symbol) const;
  [[nodiscard]] Legion::ProjectionID find_key_store_projection() const;

  [[nodiscard]] Legion::IndexSpace find_color_space_for_unbound_store(
    const Variable& partition_symbol) const;

 private:
  const Operation* operation_{};
  Domain launch_domain_{};
  std::unordered_map<Variable, InternalSharedPtr<Partition>> assignments_{};
  std::unordered_map<Variable, std::pair<Legion::FieldSpace, Legion::FieldID>>
    fields_for_unbound_stores_{};
  std::unordered_map<Variable, Legion::IndexSpace> color_spaces_for_unbound_stores_{};
  std::unordered_map<Variable, Legion::ProjectionID> projection_ids_{};
  std::optional<Variable> key_partition_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/strategy.inl>
