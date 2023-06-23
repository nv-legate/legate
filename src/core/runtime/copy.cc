/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/runtime/operation.h"

#include "core/data/detail/logical_store.h"
#include "core/data/logical_store.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/constraint_solver.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/detail/copy_launcher.h"
#include "core/runtime/detail/projection.h"

namespace legate {

Copy::Copy(int64_t unique_id, mapping::MachineDesc&& machine)
  : Operation(unique_id, std::move(machine))
{
}

void Copy::add_store(std::vector<StoreArg>& store_args,
                     LogicalStore& store,
                     const Variable* partition_symbol)
{
  auto store_impl = store.impl();
  store_args.push_back(StoreArg(store_impl.get(), partition_symbol));
  record_partition(partition_symbol, std::move(store_impl));
}

void Copy::add_store(std::optional<StoreArg>& store_arg,
                     LogicalStore& store,
                     const Variable* partition_symbol)
{
  auto store_impl = store.impl();
  store_arg       = StoreArg(store_impl.get(), partition_symbol);
  record_partition(partition_symbol, std::move(store_impl));
}

void check_store(LogicalStore store)
{
  if (store.unbound() || store.impl()->has_scalar_storage() || store.transformed()) {
    std::string msg = "Copy accepts only normal, not transformed, region-backed store";
    throw std::runtime_error(msg);
  }
}

void Copy::add_input(LogicalStore store)
{
  check_store(store);
  add_store(inputs_, store, declare_partition());
}

void Copy::add_output(LogicalStore store)
{
  check_store(store);
  if (reductions_.size() > 0)
    throw std::runtime_error("Copy targets must be either all normal outputs or reductions");
  add_store(outputs_, store, declare_partition());
}

void Copy::add_reduction(LogicalStore store, Legion::ReductionOpID redop)
{
  check_store(store);
  if (outputs_.size() > 0)
    throw std::runtime_error("Copy targets must be either all normal outputs or reductions");
  add_store(reductions_, store, declare_partition());
  reduction_ops_.push_back(redop);
}

void Copy::add_source_indirect(LogicalStore store)
{
  check_store(store);
  add_store(source_indirect_, store, declare_partition());
}

void Copy::add_target_indirect(LogicalStore store)
{
  check_store(store);
  add_store(target_indirect_, store, declare_partition());
}

void Copy::set_source_indirect_out_of_range(bool flag) { source_indirect_out_of_range_ = flag; }

void Copy::set_target_indirect_out_of_range(bool flag) { target_indirect_out_of_range_ = flag; }

void Copy::launch(detail::Strategy* p_strategy)
{
  auto& strategy = *p_strategy;
  detail::CopyLauncher launcher(
    machine_, source_indirect_out_of_range_, target_indirect_out_of_range_);
  auto launch_domain = strategy.launch_domain(this);

  auto create_projection_info = [&strategy, &launch_domain](auto& store, auto& var) {
    auto store_partition = store->create_partition(strategy[var]);
    auto proj_info       = store_partition->create_projection_info(launch_domain);
    proj_info->tag       = strategy.is_key_partition(var) ? LEGATE_CORE_KEY_STORE_TAG : 0;
    return std::move(proj_info);
  };

  for (auto& [store, var] : inputs_) launcher.add_input(store, create_projection_info(store, var));
  // FIXME: today a copy is a scatter copy only when a target indirection
  // is given. In the future, we may pass store transforms directly to
  // Legion and some transforms turn some copies into scatter copies even
  // when no target indirection is provided. So, this scatter copy check
  // will need to be extended accordingly.
  bool scatter = target_indirect_.has_value();
  for (auto& [store, var] : outputs_) {
    if (scatter)
      launcher.add_inout(store, create_projection_info(store, var));
    else
      launcher.add_output(store, create_projection_info(store, var));
    store->set_key_partition(machine(), strategy[var].get());
  }
  uint32_t idx = 0;
  for (auto& [store, var] : reductions_) {
    auto store_partition = store->create_partition(strategy[var]);
    auto proj            = store_partition->create_projection_info(launch_domain);
    bool read_write      = store_partition->is_disjoint_for(launch_domain);
    auto redop           = reduction_ops_[idx++];
    proj->set_reduction_op(redop);
    launcher.add_reduction(store, std::move(proj), read_write);
  }

  if (source_indirect_.has_value()) {
    auto& [store, var] = source_indirect_.value();
    launcher.add_source_indirect(store, create_projection_info(store, var));
  }

  if (target_indirect_.has_value()) {
    auto& [store, var] = target_indirect_.value();
    launcher.add_target_indirect(store, create_projection_info(store, var));
  }

  if (launch_domain != nullptr) {
    return launcher.execute(*launch_domain);
  } else {
    return launcher.execute_single();
  }
}

void Copy::add_to_solver(detail::ConstraintSolver& solver)
{
  bool gather  = source_indirect_.has_value();
  bool scatter = target_indirect_.has_value();

  if (inputs_.size() != outputs_.size())
    throw std::runtime_error(
      "Number of inputs and outputs should be the same in the Copy operation");

  if (gather && inputs_.size() != 1)
    throw std::runtime_error("when source indirect is specified, there could be only one input");

  if (scatter && outputs_.size() != 1)
    throw std::runtime_error("when target indirect is specified, there could be only one output");

  // fill constraints
  if (!gather && !scatter) {
    for (size_t i = 0; i < inputs_.size(); i++) {
      const auto& [in, in_part]   = inputs_[i];
      const auto& [out, out_part] = outputs_[i];
      if (in->extents() != out->extents())
        throw std::runtime_error(
          "Each output must have the same shape as the corresponding input in a copy");
      constraints_.push_back(align(in_part, out_part));
    }
  } else if (gather && scatter) {
    const auto& [src_indirect, src_indirect_part] = source_indirect_.value();
    const auto& [tgt_indirect, tgt_indirect_part] = target_indirect_.value();
    if (src_indirect->extents() != tgt_indirect->extents())
      throw std::runtime_error(
        "Source and target indirect must have the same shape in a gather-scatter copy");
    constraints_.push_back(align(src_indirect_part, tgt_indirect_part));
  } else if (gather) {
    const auto& [out, out_part] = reductions_.empty() ? outputs_.front() : reductions_.front();
    const auto& [indirect, indirect_part] = source_indirect_.value();
    if (out->extents() != indirect->extents())
      throw std::runtime_error(
        "Source indirect must have the same shape as the output in a gather copy");
    constraints_.push_back(align(out_part, indirect_part));
  } else if (scatter) {
    const auto& [in, in_part]             = inputs_.front();
    const auto& [indirect, indirect_part] = target_indirect_.value();
    if (in->extents() != indirect->extents())
      throw std::runtime_error(
        "Target indirect must have the same shape as the input in a scatter copy");
    constraints_.push_back(align(in_part, indirect_part));
  }

  for (auto& constraint : constraints_) solver.add_constraint(constraint.get());
  for (auto& [_, symb] : inputs_) solver.add_partition_symbol(symb);
  for (auto& [_, symb] : outputs_) solver.add_partition_symbol(symb, true);
  for (auto& [_, symb] : reductions_) solver.add_partition_symbol(symb);
  if (source_indirect_.has_value()) solver.add_partition_symbol(source_indirect_.value().second);
  if (target_indirect_.has_value()) solver.add_partition_symbol(target_indirect_.value().second);
}

std::string Copy::to_string() const { return "Copy:" + std::to_string(unique_id_); }

}  // namespace legate
