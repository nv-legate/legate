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

Operation::Operation(std::uint64_t unique_id,
                     std::int32_t priority,
                     mapping::detail::Machine machine)
  : unique_id_{unique_id},
    priority_{priority},
    provenance_{Runtime::get_runtime()->get_provenance()},
    machine_{std::move(machine)}
{
}

const Variable* Operation::find_or_declare_partition(InternalSharedPtr<LogicalStore> store)
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
    .emplace_back(std::make_unique<Variable>(this, static_cast<std::int32_t>(next_part_id_++)))
    .get();
}

const InternalSharedPtr<LogicalStore>& Operation::find_store(const Variable* variable) const
{
  return store_mappings_.at(*variable);
}

void Operation::record_partition(const Variable* variable, InternalSharedPtr<LogicalStore> store)
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

std::unique_ptr<StoreProjection> Operation::create_store_projection(const Strategy& strategy,
                                                                    const Domain& launch_domain,
                                                                    const StoreArg& arg)
{
  auto store_partition = create_store_partition(arg.store, strategy[arg.variable]);
  auto store_proj      = store_partition->create_store_projection(launch_domain);
  store_proj->is_key   = strategy.is_key_partition(arg.variable);
  return store_proj;
}

}  // namespace legate::detail
