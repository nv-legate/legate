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

#include "core/task/task_context.h"

#include "core/mapping/detail/mapping.h"
#include "core/task/detail/task_context.h"

namespace legate {

namespace {

std::vector<PhysicalArray> to_arrays(
  const std::vector<InternalSharedPtr<detail::PhysicalArray>>& array_impls)
{
  return {array_impls.begin(), array_impls.end()};
}

}  // namespace

std::int64_t TaskContext::task_id() const noexcept { return impl_->task_id(); }

LegateVariantCode TaskContext::variant_kind() const noexcept { return impl_->variant_kind(); }

PhysicalArray TaskContext::input(std::uint32_t index) const
{
  return PhysicalArray{impl_->inputs().at(index)};
}

std::vector<PhysicalArray> TaskContext::inputs() const { return to_arrays(impl_->inputs()); }

PhysicalArray TaskContext::output(std::uint32_t index) const
{
  return PhysicalArray{impl_->outputs().at(index)};
}

std::vector<PhysicalArray> TaskContext::outputs() const { return to_arrays(impl_->outputs()); }

PhysicalArray TaskContext::reduction(std::uint32_t index) const
{
  return PhysicalArray{impl_->reductions().at(index)};
}

std::vector<PhysicalArray> TaskContext::reductions() const
{
  return to_arrays(impl_->reductions());
}

const Scalar& TaskContext::scalar(std::uint32_t index) const { return impl_->scalars().at(index); }

const std::vector<Scalar>& TaskContext::scalars() const { return impl_->scalars(); }

comm::Communicator TaskContext::communicator(std::uint32_t index) const
{
  return impl_->communicators().at(index);
}

std::vector<comm::Communicator> TaskContext::communicators() const
{
  return impl_->communicators();
}

std::size_t TaskContext::num_inputs() const { return impl_->inputs().size(); }

std::size_t TaskContext::num_outputs() const { return impl_->outputs().size(); }

std::size_t TaskContext::num_reductions() const { return impl_->reductions().size(); }

std::size_t TaskContext::num_communicators() const { return impl_->communicators().size(); }

bool TaskContext::is_single_task() const { return impl_->is_single_task(); }

bool TaskContext::can_raise_exception() const { return impl_->can_raise_exception(); }

DomainPoint TaskContext::get_task_index() const { return impl_->get_task_index(); }

Domain TaskContext::get_launch_domain() const { return impl_->get_launch_domain(); }

mapping::TaskTarget TaskContext::target() const
{
  return mapping::detail::to_target(Processor::get_executing_processor().kind());
}

mapping::Machine TaskContext::machine() const { return mapping::Machine{impl_->machine()}; }

const std::string& TaskContext::get_provenance() const { return impl_->get_provenance(); }

}  // namespace legate
