/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/streaming/launch_domain_equality_checker.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/strategy.h>
#include <legate/runtime/detail/streaming/util.h>
#include <legate/utilities/detail/formatters.h>

#include <fmt/ostream.h>
#include <fmt/ranges.h>

namespace legate::detail {

std::string_view LaunchDomainEquality::name() const { return "Launch Domain Equality"; }

bool LaunchDomainEquality::is_streamable(const InternalSharedPtr<Operation>& op,
                                         const std::optional<InternalSharedPtr<Strategy>>& strategy,
                                         StreamingErrorContext* ctx)
{
  if (strategy.has_value()) {
    const auto& op_ld = strategy.value()->launch_domain(*op);

    if (op_ld.get_dim() == 1 && op_ld.get_volume() == 1) {
      // If we have a single task that only has scalar inputs and outputs and no reductions,
      // it should be safe to bypass this test. We need to do it this way
      // until we have data flow analysis that can actually catch these cases.
      // In the current streaming implementation we allow single tasks to be mapped
      // immediately in the mapper. Please see `select_tasks_to_map()`.
      const auto only_scalar_inputs = [&]() {
        return std::all_of(op->input_stores().begin(),
                           op->input_stores().end(),
                           [](const auto& store_arg) { return store_arg.store->volume() == 1; });
      };
      const auto only_scalar_outputs = [&]() {
        return std::all_of(op->output_stores().begin(),
                           op->output_stores().end(),
                           [](const auto& store_arg) { return store_arg.store->volume() == 1; });
      };
      if (op->reduction_stores().empty() && only_scalar_inputs() && only_scalar_outputs()) {
        return true;
      }
    }

    if (!launch_domain_.has_value()) {
      launch_domain_ = op_ld;
      return true;
    }

    if (op_ld != launch_domain_.value()) {
      ctx->append(
        // NOLINTNEXTLINE(bugprone-argument-comment)
        "Found unequal launch domains: operation's launch domain [{}, {}), launch "
        "domain of previous operation(s) [{}, {})",
        op_ld.lo(),
        op_ld.hi(),
        launch_domain_->lo(),
        launch_domain_->hi());
      return false;
    }
  }
  // TODO(amberhassaan): what to do for launch domain check if the op doesn't have
  // a launch domain
  return true;
}

}  // namespace legate::detail
