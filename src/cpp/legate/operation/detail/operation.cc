/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/operation.h>

#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

namespace {

[[nodiscard]] ZStringView OP_NAME(Operation::Kind kind)
{
  switch (kind) {
    case Operation::Kind::ATTACH: return "Attach";
    case Operation::Kind::AUTO_TASK: return "AutoTask";
    case Operation::Kind::COPY: return "Copy";
    case Operation::Kind::DISCARD: return "Discard";
    case Operation::Kind::EXECUTION_FENCE: return "ExecutionFence";
    case Operation::Kind::FILL: return "Fill";
    case Operation::Kind::GATHER: return "Gather";
    case Operation::Kind::INDEX_ATTACH: return "IndexAttach";
    case Operation::Kind::MANUAL_TASK: return "ManualTask";
    case Operation::Kind::MAPPING_FENCE: return "MappingFence";
    case Operation::Kind::REDUCE: return "Reduce";
    case Operation::Kind::SCATTER: return "Scatter";
    case Operation::Kind::SCATTER_GATHER: return "ScatterGather";
    case Operation::Kind::TIMING: return "Timing";
    case Operation::Kind::RELEASE_REGION_FIELD: return "ReleaseRegionField";
  }

  throw TracedException<std::invalid_argument>{"invalid operation kind"};
}

}  // namespace

Operation::Operation(std::uint64_t unique_id)
  : unique_id_{unique_id},
    provenance_{Runtime::get_runtime().get_provenance().to_string()},
    parallel_policy_{Runtime::get_runtime().scope().parallel_policy()}
{
}

Operation::Operation(std::uint64_t unique_id,
                     std::int32_t priority,
                     mapping::detail::Machine machine)
  : unique_id_{unique_id},
    priority_{priority},
    provenance_{Runtime::get_runtime().get_provenance().to_string()},
    machine_{std::move(machine)},
    parallel_policy_{Runtime::get_runtime().scope().parallel_policy()}
{
}

bool Operation::is_internal() const
{
  switch (kind()) {
    case Kind::ATTACH: [[fallthrough]];
    case Kind::DISCARD: [[fallthrough]];
    case Kind::EXECUTION_FENCE: [[fallthrough]];
    case Kind::INDEX_ATTACH: [[fallthrough]];
    case Kind::MAPPING_FENCE: [[fallthrough]];
    case Kind::RELEASE_REGION_FIELD: [[fallthrough]];
    case Kind::TIMING: {
      return true;
    }

    case Kind::AUTO_TASK: [[fallthrough]];
    case Kind::COPY: [[fallthrough]];
    case Kind::FILL: [[fallthrough]];
    case Kind::GATHER: [[fallthrough]];
    case Kind::MANUAL_TASK: [[fallthrough]];
    case Kind::REDUCE: [[fallthrough]];
    case Kind::SCATTER: [[fallthrough]];
    case Kind::SCATTER_GATHER: {
      return false;
    }
  }

  throw TracedException<std::invalid_argument>{"invalid operation kind"};
}

bool Operation::needs_partitioning() const
{
  switch (kind()) {
    case Kind::AUTO_TASK: [[fallthrough]];
    case Kind::COPY: [[fallthrough]];
    case Kind::FILL: [[fallthrough]];
    case Kind::GATHER: [[fallthrough]];
    case Kind::REDUCE: [[fallthrough]];
    case Kind::SCATTER: [[fallthrough]];
    case Kind::SCATTER_GATHER: {
      return true;
    }

    case Kind::ATTACH: [[fallthrough]];
    case Kind::DISCARD: [[fallthrough]];
    case Kind::EXECUTION_FENCE: [[fallthrough]];
    case Kind::INDEX_ATTACH: [[fallthrough]];
    case Kind::MANUAL_TASK: [[fallthrough]];
    case Kind::MAPPING_FENCE: [[fallthrough]];
    case Kind::RELEASE_REGION_FIELD: [[fallthrough]];
    case Kind::TIMING: {
      return false;
    }
  }

  throw TracedException<std::invalid_argument>{"invalid operation kind"};
}

std::string Operation::to_string(bool show_provenance) const
{
  auto result = fmt::format("{}:{}", kind(), unique_id_);

  if (!provenance().empty() && show_provenance) {
    fmt::format_to(std::back_inserter(result), "[{}]", provenance());
  }
  return result;
}

bool Operation::needs_flush() const
{
  LEGATE_ABORT("This method should have been overridden");
  return false;
}

const Variable* Operation::find_or_declare_partition(const InternalSharedPtr<LogicalStore>& store)
{
  const auto [it, inserted] = part_mappings_.try_emplace(store);

  if (inserted) {
    try {
      it->second = declare_partition();
    } catch (...) {
      // strong exception guarantee
      part_mappings_.erase(it);
      throw;
    }
  }
  return it->second;
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
    try {
      part_mappings_.try_emplace(mapped_store, variable);
    } catch (...) {
      // strong exception guarantee
      store_mappings_.erase(it);
      throw;
    }
  } else if (mapped_store->id() != sid) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Variable {} is already assigned to another store", *variable)};
  }
}

StoreProjection Operation::create_store_projection_(const Strategy& strategy,
                                                    const Domain& launch_domain,
                                                    const StoreArg& arg)
{
  auto store_partition = create_store_partition(arg.store, strategy[arg.variable]);
  auto store_proj      = store_partition->create_store_projection(launch_domain);

  store_proj.is_key = strategy.is_key_partition(arg.variable);
  return store_proj;
}

}  // namespace legate::detail

namespace fmt {

format_context::iterator formatter<legate::detail::Operation::Kind>::format(
  legate::detail::Operation::Kind kind, format_context& ctx) const
{
  return formatter<legate::detail::ZStringView>::format(legate::detail::OP_NAME(kind), ctx);
}

}  // namespace fmt
