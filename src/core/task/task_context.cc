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

#include "core/task/task_context.h"

#include "core/mapping/detail/mapping.h"
#include "core/runtime/detail/runtime.h"
#include "core/task/detail/task_context.h"

#include <type_traits>

#define CHECK_IF_CAN_USE_MEMBER_FUNCTION(op)                 \
  static_assert(!std::is_lvalue_reference_v<decltype(op())>, \
                "Can use this->" LEGATE_STRINGIZE(op) "() instead")

namespace legate {

namespace {

[[nodiscard]] std::vector<PhysicalArray> to_arrays(
  const std::vector<InternalSharedPtr<detail::PhysicalArray>>& array_impls)
{
  return {array_impls.begin(), array_impls.end()};
}

}  // namespace

std::int64_t TaskContext::task_id() const noexcept { return impl()->task_id(); }

LegateVariantCode TaskContext::variant_kind() const noexcept { return impl()->variant_kind(); }

PhysicalArray TaskContext::input(std::uint32_t index) const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(inputs);
  return PhysicalArray{impl()->inputs().at(index)};
}

std::vector<PhysicalArray> TaskContext::inputs() const { return to_arrays(impl()->inputs()); }

PhysicalArray TaskContext::output(std::uint32_t index) const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(outputs);
  return PhysicalArray{impl()->outputs().at(index)};
}

std::vector<PhysicalArray> TaskContext::outputs() const { return to_arrays(impl()->outputs()); }

PhysicalArray TaskContext::reduction(std::uint32_t index) const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(reductions);
  return PhysicalArray{impl()->reductions().at(index)};
}

std::vector<PhysicalArray> TaskContext::reductions() const
{
  return to_arrays(impl()->reductions());
}

const Scalar& TaskContext::scalar(std::uint32_t index) const
{
  // Revert back to using impl()->scalars() if scalars() ever returns a non-ref. No point in
  // creating a whole temporary vector just to peek at one of them :)
  static_assert(std::is_lvalue_reference_v<decltype(scalars())>);
  return scalars().at(index);
}

const std::vector<Scalar>& TaskContext::scalars() const { return impl()->scalars(); }

const comm::Communicator& TaskContext::communicator(std::uint32_t index) const
{
  // Revert back to using impl()->communicators() if communicators() ever returns a non-ref. No
  // point in creating a whole temporary vector just to check its size :)
  static_assert(std::is_lvalue_reference_v<decltype(communicators())>);
  return communicators().at(index);
}

const std::vector<comm::Communicator>& TaskContext::communicators() const
{
  return impl()->communicators();
}

std::size_t TaskContext::num_inputs() const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(inputs);
  return impl()->inputs().size();
}

std::size_t TaskContext::num_outputs() const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(outputs);
  return impl()->outputs().size();
}

std::size_t TaskContext::num_reductions() const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(reductions);
  return impl()->reductions().size();
}

std::size_t TaskContext::num_communicators() const
{
  static_assert(std::is_lvalue_reference_v<decltype(communicators())>);
  return communicators().size();
}

bool TaskContext::is_single_task() const { return impl()->is_single_task(); }

bool TaskContext::can_raise_exception() const { return impl()->can_raise_exception(); }

const DomainPoint& TaskContext::get_task_index() const { return impl()->get_task_index(); }

const Domain& TaskContext::get_launch_domain() const { return impl()->get_launch_domain(); }

mapping::TaskTarget TaskContext::target() const
{
  return mapping::detail::to_target(
    detail::Runtime::get_runtime()->get_executing_processor().kind());
}

mapping::Machine TaskContext::machine() const { return mapping::Machine{impl()->machine()}; }

std::string_view TaskContext::get_provenance() const { return impl()->get_provenance(); }

}  // namespace legate
