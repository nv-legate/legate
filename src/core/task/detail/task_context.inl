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

#pragma once

#include "core/task/detail/task_context.h"

#include <utility>

namespace legate::detail {

inline std::vector<std::shared_ptr<PhysicalArray>>& TaskContext::inputs() { return inputs_; }

inline std::vector<std::shared_ptr<PhysicalArray>>& TaskContext::outputs() { return outputs_; }

inline std::vector<std::shared_ptr<PhysicalArray>>& TaskContext::reductions()
{
  return reductions_;
}

inline const std::vector<legate::Scalar>& TaskContext::scalars() { return scalars_; }

inline std::vector<comm::Communicator>& TaskContext::communicators() { return comms_; }

inline int64_t TaskContext::task_id() const noexcept { return task_->task_id; }

inline LegateVariantCode TaskContext::variant_kind() const noexcept { return variant_kind_; }

inline bool TaskContext::is_single_task() const { return !task_->is_index_space; }

inline bool TaskContext::can_raise_exception() const { return can_raise_exception_; }

inline DomainPoint TaskContext::get_task_index() const { return task_->index_point; }

inline Domain TaskContext::get_launch_domain() const { return task_->index_domain; }

inline void TaskContext::set_exception(std::string what) { excn_ = std::move(what); }

inline std::optional<std::string>& TaskContext::get_exception() noexcept { return excn_; }

inline const mapping::detail::Machine& TaskContext::machine() const { return machine_; }

inline const std::string& TaskContext::get_provenance() const
{
  return task_->get_provenance_string();
}

}  // namespace legate::detail
