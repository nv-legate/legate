/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/fill.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/physical_store.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/dispatch.h>

#include <stdexcept>

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
  record_partition_(lhs_var_, lhs_);
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
  record_partition_(lhs_var_, lhs_);
}

void Fill::validate()
{
  // Would use const InternalSharedPtr<Type>& (since that's what these member functions
  // return), but GCC balks:
  //
  // legate/src/cpp/legate/operation/detail/fill.cc: In member
  // function 'virtual void legate::detail::Fill::validate()':
  // legate/src/cpp/legate/operation/detail/fill.cc:64:19: error:
  // possibly dangling reference to a temporary [-Werror=dangling-reference]
  //    64 |   if (const auto& value_type = std::visit(Visitor{}, value_); *lhs_->type() != ...
  //       |                   ^~~~~~~~~~
  //
  // So return by value it is...
  constexpr auto vis =
    Overload{[](const InternalSharedPtr<LogicalStore>& store) -> InternalSharedPtr<Type> {
               return store->type();
             },
             [](const Scalar& scalar) -> InternalSharedPtr<Type> { return scalar.type(); }};

  if (const auto value_type = std::visit(vis, value_); *lhs_->type() != *value_type) {
    throw TracedException<std::invalid_argument>{"Fill value and target must have the same type"};
  }
}

namespace {

class InlineFill {
 public:
  template <Type::Code CODE, std::int32_t DIM>
  void operator()(Span<const std::byte> fill_value, legate::PhysicalStore* store) const
  {
    const auto mdspan = store->template span_write_accessor<type_of_t<CODE>, DIM>();

    for_each_in_extent(mdspan.extents(), [&](auto... indices) {
      std::memcpy(static_cast<void*>(&mdspan(indices...)),
                  static_cast<const void*>(fill_value.data()),
                  fill_value.size());
    });
  }
};

}  // namespace

void Fill::launch(Strategy* /*strategy*/)
{
  LEGATE_CHECK(value_.index() == 1);
  auto&& fill_value = std::get<1>(value_);

  // This is so incredibly unclean, having to reach into the storage to do this, but it's
  // seemingly the only way to do this. We can't convert the DeferredBuffer to a logical region
  // without enormous effort, not to mention it goes through Legion, which we want to avoid for
  // the fast-path.
  auto phys_store       = lhs_->get_physical_store(mapping::StoreTarget::SYSMEM);
  auto pub_phys_store   = legate::PhysicalStore{phys_store};
  std::size_t fut_size  = 0;
  const auto* const buf = static_cast<const std::byte*>(fill_value.data());

  double_dispatch(std::max(1, static_cast<std::int32_t>(phys_store->dim())),
                  phys_store->type()->code,
                  InlineFill{},
                  Span<const std::byte>{buf, fut_size},
                  &pub_phys_store);

  lhs_->set_key_partition(machine(), parallel_policy(), create_no_partition());
}

void Fill::add_to_solver(ConstraintSolver& solver)
{
  solver.add_partition_symbol(lhs_var_, AccessMode::WRITE);
}

}  // namespace legate::detail
