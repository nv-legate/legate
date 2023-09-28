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

#include <memory>

#include "core/data/detail/logical_store.h"
#include "core/mapping/detail/machine.h"
#include "core/operation/detail/projection.h"

namespace legate::detail {
struct ConstraintSolver;
class LogicalStore;
class Strategy;
class Variable;

class Operation {
 protected:
  struct StoreArg {
    LogicalStore* store;
    const Variable* variable;
  };
  Operation(uint64_t unique_id, mapping::detail::Machine&& machine);

 public:
  virtual ~Operation() {}

 public:
  virtual void validate()                              = 0;
  virtual void add_to_solver(ConstraintSolver& solver) = 0;
  virtual void launch(Strategy* strategy)              = 0;
  virtual std::string to_string() const                = 0;

 public:
  const Variable* find_or_declare_partition(std::shared_ptr<LogicalStore> store);
  const Variable* declare_partition();
  std::shared_ptr<LogicalStore> find_store(const Variable* variable) const;

 public:
  const mapping::detail::Machine& machine() const { return machine_; }
  const std::string& provenance() const { return provenance_; }

 protected:
  void record_partition(const Variable* variable, std::shared_ptr<LogicalStore> store);
  // Helper methods
  std::unique_ptr<ProjectionInfo> create_projection_info(const Strategy& strategy,
                                                         const Domain& launch_domain,
                                                         const StoreArg& arg) const;

 protected:
  uint64_t unique_id_;
  uint32_t next_part_id_{0};
  std::vector<std::unique_ptr<Variable>> partition_symbols_{};
  std::map<const Variable, std::shared_ptr<LogicalStore>> store_mappings_{};
  std::map<std::shared_ptr<LogicalStore>, const Variable*> part_mappings_{};
  std::string provenance_;
  mapping::detail::Machine machine_;
};

}  // namespace legate::detail
