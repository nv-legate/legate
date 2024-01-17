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

Fill::Fill(InternalSharedPtr<LogicalStore>&& lhs,
           InternalSharedPtr<LogicalStore>&& value,
           uint64_t unique_id,
           mapping::detail::Machine&& machine)
  : Operation{unique_id, std::move(machine)},
    lhs_var_{declare_partition()},
    lhs_{std::move(lhs)},
    value_{std::move(value)}
{
  store_mappings_[*lhs_var_] = lhs_;
  if (lhs_->unbound()) {
    throw std::invalid_argument("Fill lhs must be a normal store");
  }

  if (!value_->has_scalar_storage()) {
    throw std::invalid_argument("Fill value should be a Future-back store");
  }
}

void Fill::validate()
{
  if (*lhs_->type() != *value_->type()) {
    throw std::invalid_argument("Fill value and target must have the same type");
  }
}

void Fill::launch(Strategy* strategy)
{
  if (lhs_->has_scalar_storage()) {
    lhs_->set_future(value_->get_future());
    return;
  }

  auto launcher      = FillLauncher{machine_};
  auto launch_domain = strategy->launch_domain(this);
  auto part          = (*strategy)[lhs_var_];
  auto lhs_proj      = create_store_partition(lhs_, part)->create_store_projection(launch_domain);

  lhs_->set_key_partition(machine(), part.get());

  if (launch_domain.is_valid()) {
    launcher.launch(launch_domain, lhs_.get(), *lhs_proj, value_.get());
  } else {
    launcher.launch_single(lhs_.get(), *lhs_proj, value_.get());
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
