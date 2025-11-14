/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_arrays/struct_logical_array.h>

#include <legate/data/detail/physical_array.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace legate::detail {

std::uint32_t StructLogicalArray::dim() const { return fields_.front()->dim(); }

const InternalSharedPtr<Shape>& StructLogicalArray::shape() const
{
  return fields_.front()->shape();
}

std::size_t StructLogicalArray::volume() const { return fields_.front()->volume(); }

bool StructLogicalArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto& array) { return array->unbound(); });
}

namespace {

// Some API still take std::vector, so provide a copy of `make_array_from_op()` that produces
// `std::vector`s.
template <typename U, typename T, typename F>
std::vector<U> make_vector_from_op(Span<const T> fields, F&& generator_fn)
{
  std::vector<U> result;

  result.reserve(fields.size());
  std::transform(
    fields.begin(), fields.end(), std::back_inserter(result), std::forward<F>(generator_fn));
  return result;
}

// provide a generic template in case the return type of the vector is different from
// fields/needs to be manually specified
template <typename U, typename T, typename F>
SmallVector<U> make_array_from_op(Span<const T> fields, F&& generator_fn)
{
  SmallVector<U> result;

  result.reserve(fields.size());
  std::transform(
    fields.begin(), fields.end(), std::back_inserter(result), std::forward<F>(generator_fn));
  return result;
}

// in case the return type of the vector is the same as fields
template <typename T, typename F>
SmallVector<T> make_array_from_op(Span<const T> fields, F&& generator_fn)
{
  return make_array_from_op<T, T, F>(fields, std::forward<F>(generator_fn));
}

}  // namespace

bool StructLogicalArray::is_mapped() const
{
  return (nullable() && null_mask()->is_mapped()) ||
         std::any_of(
           fields().begin(), fields().end(), [](auto&& field) { return field->is_mapped(); });
}

InternalSharedPtr<LogicalArray> StructLogicalArray::promote(std::int32_t extra_dim,
                                                            std::size_t dim_size) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->promote(extra_dim, dim_size)) : std::nullopt;
  auto fields = make_array_from_op(
    this->fields(), [&](auto& field) { return field->promote(extra_dim, dim_size); });

  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::project(std::int32_t dim,
                                                            std::int64_t index) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->project(dim, index)) : std::nullopt;
  auto fields =
    make_array_from_op(this->fields(), [&](auto& field) { return field->project(dim, index); });

  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::broadcast(std::int32_t dim,
                                                              std::size_t dim_size) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->broadcast(dim, dim_size)) : std::nullopt;
  auto fields = make_array_from_op(this->fields(),
                                   [&](auto& field) { return field->broadcast(dim, dim_size); });

  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::slice(std::int32_t dim, Slice sl) const
{
  auto null_mask =
    nullable() ? std::make_optional(slice_store(this->null_mask(), dim, sl)) : std::nullopt;
  auto fields =
    make_array_from_op(this->fields(), [&](auto& field) { return field->slice(dim, sl); });
  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::transpose(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> axes) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->transpose(axes)) : std::nullopt;
  auto fields =
    make_array_from_op(this->fields(), [&](auto& field) { return field->transpose(axes); });
  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::delinearize(
  std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->delinearize(dim, sizes)) : std::nullopt;
  auto fields =
    make_array_from_op(this->fields(), [&](auto& field) { return field->delinearize(dim, sizes); });
  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

const InternalSharedPtr<LogicalStore>& StructLogicalArray::null_mask() const
{
  if (!nullable()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve the null mask of a non-nullable array"};
  }
  return *null_mask_;  // NOLINT(bugprone-unchecked-optional-access)
}

InternalSharedPtr<PhysicalArray> StructLogicalArray::get_physical_array(
  legate::mapping::StoreTarget target, bool ignore_future_mutability) const
{
  std::optional<InternalSharedPtr<PhysicalStore>> null_mask_store{};

  if (nullable()) {
    null_mask_store = null_mask()->get_physical_store(target, ignore_future_mutability);
  }

  auto field_arrays = make_array_from_op<InternalSharedPtr<PhysicalArray>>(
    fields(),
    [&](auto& field) { return field->get_physical_array(target, ignore_future_mutability); });

  return make_internal_shared<StructPhysicalArray>(
    type_, std::move(null_mask_store), std::move(field_arrays));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::child(std::uint32_t index) const
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve a sub-array of an unbound array"};
  }
  return fields_.at(index);
}

const InternalSharedPtr<LogicalStore>& StructLogicalArray::primary_store() const
{
  return fields_.front()->primary_store();
}

void StructLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  for (auto&& field : fields_) {
    field->record_scalar_or_unbound_outputs(task);
  }

  if (!nullable()) {
    return;
  }

  if (null_mask()->unbound()) {
    task->record_unbound_output(null_mask());
  } else if (null_mask()->has_scalar_storage()) {
    task->record_scalar_output(null_mask());
  }
}

void StructLogicalArray::record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const
{
  for (auto&& field : fields_) {
    field->record_scalar_reductions(task, redop);
  }
  if (nullable() && null_mask()->has_scalar_storage()) {
    auto null_redop = bool_()->find_reduction_operator(ReductionOpKind::MUL);
    task->record_scalar_reduction(null_mask(), null_redop);
  }
}

void StructLogicalArray::generate_constraints(
  AutoTask* task,
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  auto it = fields_.begin();
  (*it++)->generate_constraints(task, mapping, partition_symbol);
  for (; it != fields_.end(); ++it) {
    auto part_field = task->declare_partition();
    (*it)->generate_constraints(task, mapping, part_field);
    // Need to bypass the signature check here because these generated constraints are not
    // technically visible to the user (you cannot declare different constraints on the "main"
    // store and the nullable store in the signature).
    task->add_constraint(align(partition_symbol, part_field), /* bypass_signature_check */ true);
  }

  if (!nullable()) {
    return;
  }
  auto part_null_mask = task->declare_partition();
  mapping.try_emplace(null_mask(), part_null_mask);
  // Need to bypass the signature check here because these generated constraints are not
  // technically visible to the user (you cannot declare different constraints on the "main"
  // store and the nullable store in the signature).
  task->add_constraint(align(partition_symbol, part_null_mask), /* bypass_signature_check */ true);
}

ArrayAnalyzable StructLogicalArray::to_launcher_arg(
  const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  GlobalRedopID redop) const
{
  std::optional<StoreAnalyzable> null_mask_arg;

  if (nullable()) {
    auto null_redop = privilege == LEGION_REDUCE
                        ? bool_()->find_reduction_operator(ReductionOpKind::MUL)
                        : GlobalRedopID{-1};
    null_mask_arg   = store_to_launcher_arg(null_mask(),
                                          mapping.at(null_mask()),
                                          strategy,
                                          launch_domain,
                                          projection,
                                          privilege,
                                          null_redop);
  }

  auto field_args = make_vector_from_op<ArrayAnalyzable>(fields(), [&](auto& field) {
    return field->to_launcher_arg(mapping, strategy, launch_domain, projection, privilege, redop);
  });

  return StructArrayArg{type(), std::move(null_mask_arg), std::move(field_args)};
}

ArrayAnalyzable StructLogicalArray::to_launcher_arg_for_fixup(const Domain& launch_domain,
                                                              Legion::PrivilegeMode privilege) const
{
  return StructArrayArg{
    type(), std::nullopt, make_vector_from_op<ArrayAnalyzable>(fields(), [&](auto& field) {
      return field->to_launcher_arg_for_fixup(launch_domain, privilege);
    })};
}

void StructLogicalArray::collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const
{
  if (nullable()) {
    trackers.emplace_back(null_mask());
  }
  for (auto&& field : fields_) {
    field->collect_storage_trackers(trackers);
  }
}

void StructLogicalArray::calculate_pack_size(TaskReturnLayoutForUnpack* layout) const
{
  if (nullable()) {
    null_mask()->calculate_pack_size(layout);
  }
  for (auto&& field : fields_) {
    field->calculate_pack_size(layout);
  }
}

}  // namespace legate::detail
