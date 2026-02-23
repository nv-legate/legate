/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/fill.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/physical_store.h>
#include <legate/operation/detail/fill_launcher.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/partitioning/detail/constraint_solver.h>
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
  record_partition_(lhs_var_, lhs_, AccessMode::WRITE);
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
  record_partition_(lhs_var_, lhs_, AccessMode::WRITE);
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

void Fill::launch(Strategy* strategy)
{
  auto&& fill_value = get_fill_value_();
  if (lhs_->has_scalar_storage()) {
    lhs_->set_future(std::move(fill_value));
    return;
  }

  // This is so incredibly unclean, having to reach into the storage to do this, but it's
  // seemingly the only way to do this. We can't convert the DeferredBuffer to a logical region
  // without enormous effort, not to mention it goes through Legion, which we want to avoid for
  // the fast-path.
  if (lhs_->get_storage()->kind() == Storage::Kind::INLINE_STORAGE) {
    auto phys_store =
      lhs_->get_physical_store(mapping::StoreTarget::SYSMEM, /*ignore_future_mutability=*/false);
    auto pub_phys_store   = legate::PhysicalStore{phys_store};
    std::size_t fut_size  = 0;
    const auto* const buf = static_cast<const std::byte*>(
      fill_value.get_buffer(find_memory_kind_for_executing_processor(), &fut_size));

    double_dispatch(std::max(1, static_cast<std::int32_t>(phys_store->dim())),
                    phys_store->type()->code,
                    InlineFill{},
                    Span<const std::byte>{buf, fut_size},
                    &pub_phys_store);
    return;
  }

  auto launcher        = FillLauncher{machine_, priority(), provenance().as_string_view()};
  auto&& launch_domain = strategy->launch_domain(*this);
  auto&& part          = (*strategy)[*lhs_var_];
  const auto lhs_proj  = create_store_partition(lhs_, part)->create_store_projection(launch_domain);

  if (launch_domain.is_valid()) {
    launcher.launch(launch_domain, lhs_.get(), lhs_proj, std::move(fill_value));
    lhs_->set_key_partition(machine(), parallel_policy(), part);
  } else {
    launcher.launch_single(lhs_.get(), lhs_proj, std::move(fill_value));
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

  return std::visit(
    Overload{[](const InternalSharedPtr<LogicalStore>& store) { return store->needs_flush(); },
             [](const Scalar&) { return false; }},
    value_);
}

Legion::Future Fill::get_fill_value_() const
{
  return std::visit(
    Overload{[](const InternalSharedPtr<LogicalStore>& store) { return store->get_future(); },
             [](const Scalar& scalar) {
               return Legion::Future::from_untyped_pointer(scalar.data(), scalar.size());
             }},
    value_);
}

}  // namespace legate::detail
