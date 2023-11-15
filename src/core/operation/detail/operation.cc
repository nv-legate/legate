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

#include "core/operation/detail/operation.h"

#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

Operation::Operation(uint64_t unique_id, mapping::detail::Machine&& machine)
  : unique_id_{unique_id},
    provenance_{Runtime::get_runtime()->provenance_manager()->get_provenance()},
    machine_{std::move(machine)}
{
}

const Variable* Operation::find_or_declare_partition(std::shared_ptr<LogicalStore> store)
{
  auto finder = part_mappings_.find(store);
  if (finder != part_mappings_.end()) {
    return finder->second;
  }
  const auto* symb                 = declare_partition();
  part_mappings_[std::move(store)] = symb;
  return symb;
}

const Variable* Operation::declare_partition()
{
  return partition_symbols_
    .emplace_back(std::make_unique<Variable>(this, static_cast<int32_t>(next_part_id_++)))
    .get();
}

std::shared_ptr<LogicalStore> Operation::find_store(const Variable* variable) const
{
  return store_mappings_.at(*variable);
}

void Operation::record_partition(const Variable* variable, std::shared_ptr<LogicalStore> store)
{
  auto finder = store_mappings_.find(*variable);
  if (finder != store_mappings_.end()) {
    if (finder->second->id() != store->id()) {
      throw std::invalid_argument("Variable " + variable->to_string() +
                                  " is already assigned to another store");
    }
    return;
  }
  if (part_mappings_.find(store) == part_mappings_.end()) {
    part_mappings_.insert({store, variable});
  }
  store_mappings_[*variable] = std::move(store);
}

std::unique_ptr<ProjectionInfo> Operation::create_projection_info(const Strategy& strategy,
                                                                  const Domain& launch_domain,
                                                                  const StoreArg& arg)
{
  auto store_partition = arg.store->create_partition(strategy[arg.variable]);
  auto proj_info       = store_partition->create_projection_info(launch_domain);
  proj_info->is_key    = strategy.is_key_partition(arg.variable);
  return proj_info;
}

}  // namespace legate::detail
