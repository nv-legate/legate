/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fmt/format.h>
#include <stdexcept>
#include <string_view>

namespace legate::detail {

namespace {

[[nodiscard]] std::string_view OP_NAME(Operation::Kind kind)
{
  switch (kind) {
    case Operation::Kind::AUTO_TASK: return "AutoTask";
    case Operation::Kind::COPY: return "Copy";
    case Operation::Kind::FILL: return "Fill";
    case Operation::Kind::GATHER: return "Gather";
    case Operation::Kind::MANUAL_TASK: return "ManualTask";
    case Operation::Kind::REDUCE: return "Reduce";
    case Operation::Kind::SCATTER: return "Scatter";
    case Operation::Kind::SCATTER_GATHER: return "ScatterGather";
  }

  throw std::invalid_argument{"invalid operation kind"};
}

}  // namespace

Operation::Operation(std::uint64_t unique_id,
                     std::int32_t priority,
                     mapping::detail::Machine machine)
  : unique_id_{unique_id},
    priority_{priority},
    provenance_{Runtime::get_runtime()->get_provenance()},
    machine_{std::move(machine)}
{
}

std::string Operation::to_string() const
{
  auto result = fmt::format("{}:{}", kind(), unique_id_);

  if (!provenance().empty()) {
    fmt::format_to(std::back_inserter(result), "[{}]", provenance());
  }
  return result;
}

const Variable* Operation::find_or_declare_partition(const InternalSharedPtr<LogicalStore>& store)
{
  if (const auto it = part_mappings_.find(store); it != part_mappings_.end()) {
    return it->second;
  }
  const auto* symb      = declare_partition();
  part_mappings_[store] = symb;
  return symb;
}

const Variable* Operation::declare_partition()
{
  return &partition_symbols_.emplace_back(this, next_part_id_++);
}

const InternalSharedPtr<LogicalStore>& Operation::find_store(const Variable* variable) const
{
  return store_mappings_.at(*variable);
}

void Operation::record_partition_(
  const Variable* variable,
  // Obviously, it is moved, but clang-tidy does not see that...
  InternalSharedPtr<LogicalStore> store  // NOLINT(performance-unnecessary-value-param)
)
{
  const auto sid            = store->id();
  const auto [it, inserted] = store_mappings_.try_emplace(*variable, std::move(store));
  const auto& mapped_store  = it->second;

  if (inserted) {
    part_mappings_.try_emplace(mapped_store, variable);
  } else if (mapped_store->id() != sid) {
    throw std::invalid_argument{
      fmt::format("Variable {} is already assigned to another store", *variable)};
  }
}

std::unique_ptr<StoreProjection> Operation::create_store_projection_(const Strategy& strategy,
                                                                     const Domain& launch_domain,
                                                                     const StoreArg& arg)
{
  auto store_partition = create_store_partition(arg.store, strategy[arg.variable]);
  auto store_proj      = store_partition->create_store_projection(launch_domain);
  store_proj->is_key   = strategy.is_key_partition(arg.variable);
  return store_proj;
}

}  // namespace legate::detail

namespace fmt {

format_context::iterator formatter<legate::detail::Operation::Kind>::format(
  legate::detail::Operation::Kind kind, format_context& ctx) const
{
  return formatter<string_view>::format(legate::detail::OP_NAME(kind), ctx);
}

}  // namespace fmt
