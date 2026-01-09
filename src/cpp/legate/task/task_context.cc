/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/task_context.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/task_context.h>
#include <legate/utilities/span.h>

#include <type_traits>

#define CHECK_IF_CAN_USE_MEMBER_FUNCTION(op)                 \
  static_assert(!std::is_lvalue_reference_v<decltype(op())>, \
                "Can use this->" LEGATE_STRINGIZE(op) "() instead")

namespace legate {

namespace {

[[nodiscard]] std::vector<PhysicalArray> to_arrays(
  Span<const InternalSharedPtr<detail::PhysicalArray>> array_impls)
{
  return {array_impls.begin(), array_impls.end()};
}

}  // namespace

GlobalTaskID TaskContext::task_id() const noexcept { return impl()->task_id(); }

VariantCode TaskContext::variant_kind() const noexcept { return impl()->variant_kind(); }

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

Scalar TaskContext::scalar(std::uint32_t index) const
{
  return Scalar{impl()->scalars().at(index)};
}

std::vector<Scalar> TaskContext::scalars() const
{
  auto&& scals = impl()->scalars();
  return {scals.begin(), scals.end()};
}

const comm::Communicator& TaskContext::communicator(std::uint32_t index) const
{
  return impl()->communicators().at(index);
}

std::vector<comm::Communicator> TaskContext::communicators() const
{
  auto&& comms = impl()->communicators();

  return {comms.begin(), comms.end()};
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

std::size_t TaskContext::num_scalars() const
{
  CHECK_IF_CAN_USE_MEMBER_FUNCTION(scalars);
  return impl()->scalars().size();
}

std::size_t TaskContext::num_communicators() const { return impl()->communicators().size(); }

bool TaskContext::is_single_task() const { return impl()->is_single_task(); }

bool TaskContext::can_raise_exception() const { return impl()->can_raise_exception(); }

const DomainPoint& TaskContext::get_task_index() const { return impl()->get_task_index(); }

const Domain& TaskContext::get_launch_domain() const { return impl()->get_launch_domain(); }

mapping::TaskTarget TaskContext::target() const
{
  return mapping::detail::to_target(
    detail::Runtime::get_runtime().get_executing_processor().kind());
}

mapping::Machine TaskContext::machine() const { return mapping::Machine{impl()->machine()}; }

std::string_view TaskContext::get_provenance() const { return impl()->get_provenance(); }

// NOLINTNEXTLINE(readability-make-member-function-const)
void TaskContext::concurrent_task_barrier() { impl()->concurrent_task_barrier(); }

CUstream_st* TaskContext::get_task_stream() const { return impl()->get_task_stream(); }

}  // namespace legate
