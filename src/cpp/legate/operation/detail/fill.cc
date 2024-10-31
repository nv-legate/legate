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

#include "legate/operation/detail/fill.h"

#include "legate/data/detail/logical_store.h"
#include "legate/operation/detail/fill_launcher.h"
#include "legate/operation/detail/store_projection.h"
#include "legate/partitioning/detail/constraint_solver.h"
#include "legate/partitioning/detail/partitioner.h"

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
  auto&& fill_value = get_fill_value_();
  if (lhs_->has_scalar_storage()) {
    lhs_->set_future(std::move(fill_value));
    return;
  }

  auto launcher        = FillLauncher{machine_, priority()};
  auto&& launch_domain = strategy->launch_domain(this);
  auto&& part          = (*strategy)[lhs_var_];
  auto lhs_proj        = create_store_partition(lhs_, part)->create_store_projection(launch_domain);

  if (launch_domain.is_valid()) {
    launcher.launch(launch_domain, lhs_.get(), *lhs_proj, std::move(fill_value));
    lhs_->set_key_partition(machine(), part);
  } else {
    launcher.launch_single(lhs_.get(), *lhs_proj, std::move(fill_value));
  }
}

void Fill::add_to_solver(ConstraintSolver& solver)
{
  solver.add_partition_symbol(lhs_var_, AccessMode::WRITE);
  if (lhs_->has_scalar_storage()) {
    solver.add_constraint(broadcast(lhs_var_));
  }
}

bool Fill::needs_flush() const
{
  if (lhs_->needs_flush()) {
    return true;
  }
  if (const auto* logical_store = std::get_if<InternalSharedPtr<LogicalStore>>(&value_)) {
    return (*logical_store)->needs_flush();
  }
  return false;
}

Legion::Future Fill::get_fill_value_() const
{
  if (const auto* logical_store = std::get_if<InternalSharedPtr<LogicalStore>>(&value_)) {
    return (*logical_store)->get_future();
  }

  const auto& value = std::get<Scalar>(value_);

  return Legion::Future::from_untyped_pointer(value.data(), value.size());
}

}  // namespace legate::detail
