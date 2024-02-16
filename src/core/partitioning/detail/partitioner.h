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

#include "core/data/shape.h"
#include "core/partitioning/detail/constraint.h"
#include "core/utilities/hash.h"
#include "core/utilities/internal_shared_ptr.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace legate {
struct Partition;
}  // namespace legate

namespace legate::detail {

struct ConstraintSolver;
class Partitioner;

class Strategy {
  friend class Partitioner;

 public:
  [[nodiscard]] bool parallel(const Operation* op) const;
  [[nodiscard]] Domain launch_domain(const Operation* op) const;
  void set_launch_domain(const Operation* op, const Domain& domain);

  void insert(const Variable* partition_symbol, InternalSharedPtr<Partition> partition);
  void insert(const Variable* partition_symbol,
              InternalSharedPtr<Partition> partition,
              Legion::FieldSpace field_space,
              Legion::FieldID field_id);
  [[nodiscard]] bool has_assignment(const Variable* partition_symbol) const;
  [[nodiscard]] InternalSharedPtr<Partition> operator[](const Variable* partition_symbol) const;
  [[nodiscard]] const std::pair<Legion::FieldSpace, Legion::FieldID>& find_field_for_unbound_store(
    const Variable* partition_symbol) const;
  [[nodiscard]] bool is_key_partition(const Variable* partition_symbol) const;

  void dump() const;

 private:
  void compute_launch_domains(const ConstraintSolver& solver);
  void record_key_partition(const Variable* partition_symbol);

  std::unordered_map<Variable, InternalSharedPtr<Partition>> assignments_{};
  std::unordered_map<Variable, std::pair<Legion::FieldSpace, Legion::FieldID>>
    fields_for_unbound_stores_{};
  std::unordered_map<const Operation*, Domain> launch_domains_{};
  std::optional<const Variable*> key_partition_{};
};

class Partitioner {
 public:
  explicit Partitioner(std::vector<Operation*>&& operations);

  [[nodiscard]] std::unique_ptr<Strategy> partition_stores();

 private:
  // Populates solutions for unbound stores in the `strategy` and returns remaining partition
  // symbols
  [[nodiscard]] static std::vector<const Variable*> handle_unbound_stores(
    Strategy* strategy,
    const std::vector<const Variable*>& partition_symbols,
    const ConstraintSolver& solver);

  std::vector<Operation*> operations_{};
};

}  // namespace legate::detail

#include "core/partitioning/detail/partitioner.inl"
