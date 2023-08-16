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
#include "core/task/detail/task_context.h"

namespace legate {

namespace {

std::vector<Array> to_arrays(const std::vector<std::shared_ptr<detail::Array>>& array_impls)
{
  std::vector<Array> result;
  for (const auto& array_impl : array_impls) { result.emplace_back(array_impl); }
  return std::move(result);
}

}  // namespace

Array TaskContext::input(uint32_t index) const { return Array(impl_->inputs().at(index)); }

std::vector<Array> TaskContext::inputs() const { return to_arrays(impl_->inputs()); }

Array TaskContext::output(uint32_t index) const { return Array(impl_->outputs().at(index)); }

std::vector<Array> TaskContext::outputs() const { return to_arrays(impl_->outputs()); }

Array TaskContext::reduction(uint32_t index) const { return Array(impl_->reductions().at(index)); }

std::vector<Array> TaskContext::reductions() const { return to_arrays(impl_->reductions()); }

const Scalar& TaskContext::scalar(uint32_t index) const { return impl_->scalars().at(index); }

const std::vector<Scalar>& TaskContext::scalars() const { return impl_->scalars(); }

std::vector<comm::Communicator> TaskContext::communicators() const
{
  return impl_->communicators();
}

size_t TaskContext::num_inputs() const { return impl_->inputs().size(); }

size_t TaskContext::num_outputs() const { return impl_->outputs().size(); }

size_t TaskContext::num_reductions() const { return impl_->reductions().size(); }

bool TaskContext::is_single_task() const { return impl_->is_single_task(); }

bool TaskContext::can_raise_exception() const { return impl_->can_raise_exception(); }

DomainPoint TaskContext::get_task_index() const { return impl_->get_task_index(); }

Domain TaskContext::get_launch_domain() const { return impl_->get_launch_domain(); }

mapping::Machine TaskContext::machine() const { return mapping::Machine(impl_->machine()); }

const std::string& TaskContext::get_provenance() const { return impl_->get_provenance(); }

TaskContext::TaskContext(detail::TaskContext* impl) : impl_(impl) {}

// No need to delete impl_, as it's owned by the caller
TaskContext::~TaskContext() {}

TaskContext::TaskContext(const TaskContext&) = default;

TaskContext& TaskContext::operator=(const TaskContext&) = default;

}  // namespace legate
