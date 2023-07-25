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

std::vector<Store>& TaskContext::inputs() { return impl_->inputs(); }

std::vector<Store>& TaskContext::outputs() { return impl_->outputs(); }

std::vector<Store>& TaskContext::reductions() { return impl_->reductions(); }

std::vector<Scalar>& TaskContext::scalars() { return impl_->scalars(); }

std::vector<comm::Communicator>& TaskContext::communicators() { return impl_->communicators(); }

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
