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

#include "core/operation/detail/fill.h"

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/fill_launcher.h"
#include "core/operation/detail/store_projection.h"
#include "core/partitioning/detail/constraint_solver.h"
#include "core/partitioning/detail/partitioner.h"

namespace legate::detail {

Fill::Fill(InternalSharedPtr<LogicalStore> lhs,
           InternalSharedPtr<LogicalStore> value,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine)
  : Operation{unique_id, priority, std::move(machine)},
    lhs_var_{declare_partition()},
    lhs_{std::move(lhs)},
    value_{std::move(value)}
{
  store_mappings_[*lhs_var_] = lhs_;
}

Fill::Fill(InternalSharedPtr<LogicalStore> lhs,
           Scalar value,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine)
  : Operation{unique_id, priority, std::move(machine)},
    lhs_var_{declare_partition()},
    lhs_{std::move(lhs)},
    value_{std::move(value)}
{
  store_mappings_[*lhs_var_] = lhs_;
}

void Fill::validate()
{
  const auto& value_type =
    value_.index() == 0 ? std::get<0>(value_)->type() : std::get<1>(value_).type();
  if (*lhs_->type() != *value_type) {
    throw std::invalid_argument{"Fill value and target must have the same type"};
  }
}

void Fill::launch(Strategy* strategy)
{
  if (lhs_->has_scalar_storage()) {
    if (value_.index() == 0) {
      lhs_->set_future(std::get<0>(value_)->get_future());
    } else {
      auto& value = std::get<1>(value_);
      lhs_->set_future(Legion::Future::from_untyped_pointer(value.data(), value.size()));
    }
    return;
  }

  auto launcher      = FillLauncher{machine_, priority()};
  auto launch_domain = strategy->launch_domain(this);
  auto part          = (*strategy)[lhs_var_];
  auto lhs_proj      = create_store_partition(lhs_, part)->create_store_projection(launch_domain);

  lhs_->set_key_partition(machine(), part.get());

  if (value_.index() == 0) {
    if (launch_domain.is_valid()) {
      launcher.launch(launch_domain, lhs_.get(), *lhs_proj, std::get<0>(value_).get());
    } else {
      launcher.launch_single(lhs_.get(), *lhs_proj, std::get<0>(value_).get());
    }
  } else {
    if (launch_domain.is_valid()) {
      launcher.launch(launch_domain, lhs_.get(), *lhs_proj, std::get<1>(value_));
    } else {
      launcher.launch_single(lhs_.get(), *lhs_proj, std::get<1>(value_));
    }
  }
}

std::string Fill::to_string() const { return "Fill:" + std::to_string(unique_id_); }

void Fill::add_to_solver(ConstraintSolver& solver)
{
  solver.add_partition_symbol(lhs_var_, AccessMode::WRITE);
  if (lhs_->has_scalar_storage()) {
    solver.add_constraint(broadcast(lhs_var_));
  }
}

}  // namespace legate::detail
