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

#include "core/data/detail/logical_store.h"
#include "core/legate_c.h"
#include "core/mapping/detail/machine.h"
#include "core/operation/detail/store_projection.h"
#include "core/partitioning/detail/constraint.h"
#include "core/utilities/hash.h"
#include "core/utilities/internal_shared_ptr.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace legate::detail {
class ConstraintSolver;
class Strategy;

class Operation {
 protected:
  class StoreArg {
   public:
    InternalSharedPtr<LogicalStore> store{};
    const Variable* variable{};
  };

  Operation(std::uint64_t unique_id, std::int32_t priority, mapping::detail::Machine machine);

 public:
  enum class Kind : std::uint8_t {
    AUTO_TASK,
    COPY,
    FILL,
    GATHER,
    MANUAL_TASK,
    REDUCE,
    SCATTER,
    SCATTER_GATHER,
  };

  virtual ~Operation() = default;

  virtual void validate()                              = 0;
  virtual void add_to_solver(ConstraintSolver& solver) = 0;
  virtual void launch(Strategy* strategy)              = 0;

  [[nodiscard]] virtual Kind kind() const = 0;
  [[nodiscard]] virtual std::string to_string() const;
  [[nodiscard]] virtual bool always_flush() const;
  [[nodiscard]] virtual bool supports_replicated_write() const;

  [[nodiscard]] const Variable* find_or_declare_partition(InternalSharedPtr<LogicalStore> store);
  [[nodiscard]] const Variable* declare_partition();
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& find_store(const Variable* variable) const;

  [[nodiscard]] std::int32_t priority() const;
  [[nodiscard]] const mapping::detail::Machine& machine() const;
  [[nodiscard]] const std::string& provenance() const;

 protected:
  void record_partition(const Variable* variable, InternalSharedPtr<LogicalStore> store);
  // Helper methods
  [[nodiscard]] static std::unique_ptr<StoreProjection> create_store_projection(
    const Strategy& strategy, const Domain& launch_domain, const StoreArg& arg);

  std::uint64_t unique_id_{};
  std::uint32_t next_part_id_{};
  std::int32_t priority_{LEGATE_CORE_DEFAULT_TASK_PRIORITY};
  std::vector<std::unique_ptr<Variable>> partition_symbols_{};
  std::unordered_map<Variable, InternalSharedPtr<LogicalStore>> store_mappings_{};
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*> part_mappings_{};
  std::string provenance_{};
  mapping::detail::Machine machine_{};
};

}  // namespace legate::detail

#include "core/operation/detail/operation.inl"
