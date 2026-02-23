/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/copy.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/physical_store.h>
#include <legate/operation/detail/copy_launcher.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>
#include <utility>

namespace legate::detail {

Copy::Copy(InternalSharedPtr<LogicalStore> target,
           InternalSharedPtr<LogicalStore> source,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine,
           std::optional<std::int32_t> redop_kind)
  : Operation{unique_id, priority, std::move(machine)},
    target_{std::move(target), declare_partition()},
    source_{std::move(source), declare_partition()},
    constraint_{align(target_.variable, source_.variable)},
    redop_kind_{redop_kind}
{
  record_partition_(
    target_.variable, target_.store, redop_kind_ ? AccessMode::REDUCE : AccessMode::WRITE);
  record_partition_(source_.variable, source_.store, AccessMode::READ);
}

void Copy::validate()
{
  if (*source_.store->type() != *target_.store->type()) {
    throw TracedException<std::invalid_argument>{"Source and target must have the same type"};
  }
  constexpr auto validate_store = [](const auto& store) {
    if (store->unbound() || store->transformed()) {
      throw TracedException<std::invalid_argument>{
        "Copy accepts only normal and untransformed stores"};
    }
  };
  validate_store(target_.store);
  validate_store(source_.store);
  constraint_->validate();

  if (target_.store->has_scalar_storage() != source_.store->has_scalar_storage()) {
    throw TracedException<std::runtime_error>{
      "Copies are supported only between the same kind of stores"};
  }
  if (redop_kind_.has_value()) {
    if (target_.store->has_scalar_storage()) {
      throw TracedException<std::runtime_error>{
        "Reduction copies don't support future-backed target stores"};
    }
    // Try to retrieve the reduction operator ID here to check if it's defined for the value type
    static_cast<void>(target_.store->type()->find_reduction_operator(*redop_kind_));
  }
}

namespace {

class InlineCopy {
 public:
  template <Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& src, legate::PhysicalStore* target) const
  {
    const auto src_span  = src.template span_read_accessor<type_of_t<CODE>, DIM>();
    const auto dest_span = target->template span_write_accessor<type_of_t<CODE>, DIM>();

    for_each_in_extent(src_span.extents(),
                       [&](auto... indices) { dest_span(indices...) = src_span(indices...); });
  }
};

}  // namespace

void Copy::launch(Strategy* p_strategy)
{
  if (target_.store->has_scalar_storage()) {
    LEGATE_CHECK(source_.store->has_scalar_storage());
    target_.store->set_future(source_.store->get_future());
    return;
  }

  if (target_.store->get_storage()->kind() == Storage::Kind::INLINE_STORAGE ||
      source_.store->get_storage()->kind() == Storage::Kind::INLINE_STORAGE) {
    const auto src_phys =
      legate::PhysicalStore{source_.store->get_physical_store(mapping::StoreTarget::SYSMEM,
                                                              /*ignore_future_mutability=*/false)};
    auto target_phys =
      legate::PhysicalStore{target_.store->get_physical_store(mapping::StoreTarget::SYSMEM,
                                                              /*ignore_future_mutability=*/false)};

    double_dispatch(src_phys.dim(), src_phys.type().code(), InlineCopy{}, src_phys, &target_phys);
    return;
  }

  auto& strategy       = *p_strategy;
  auto launcher        = CopyLauncher{machine_, priority()};
  auto&& launch_domain = strategy.launch_domain(*this);

  launcher.add_input(source_.store, create_store_projection_(strategy, launch_domain, source_));

  if (!redop_kind_.has_value()) {
    launcher.add_output(target_.store, create_store_projection_(strategy, launch_domain, target_));
  } else {
    auto store_partition = create_store_partition(target_.store, strategy[*target_.variable]);
    auto proj            = store_partition->create_store_projection(launch_domain);

    proj.set_reduction_op(target_.store->type()->find_reduction_operator(*redop_kind_));
    launcher.add_reduction(target_.store, std::move(proj));
  }

  if (launch_domain.is_valid()) {
    launcher.execute(launch_domain);
    return;
  }
  launcher.execute_single();
}

void Copy::add_to_solver(ConstraintSolver& solver)
{
  // TODO(amberhassaan): do not move constraints until
  // https://github.com/nv-legate/legate.internal/issues/3120 is resolved
  // solver.add_constraint(std::move(constraint_));
  solver.add_constraint(constraint_);
  solver.add_partition_symbol(target_.variable,
                              !redop_kind_ ? AccessMode::WRITE : AccessMode::REDUCE);
  if (target_.store->has_scalar_storage()) {
    solver.add_constraint(broadcast(target_.variable));
  }
  solver.add_partition_symbol(source_.variable, AccessMode::READ);
}

bool Copy::needs_flush() const { return target_.needs_flush() || source_.needs_flush(); }

}  // namespace legate::detail
