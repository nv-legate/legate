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

#pragma once

#include "core/task/detail/task_context.h"

#include <utility>

namespace legate::detail {

inline std::vector<InternalSharedPtr<PhysicalArray>>& TaskContext::inputs() { return inputs_; }

inline std::vector<InternalSharedPtr<PhysicalArray>>& TaskContext::outputs() { return outputs_; }

inline std::vector<InternalSharedPtr<PhysicalArray>>& TaskContext::reductions()
{
  return reductions_;
}

inline const std::vector<legate::Scalar>& TaskContext::scalars() const noexcept { return scalars_; }

inline const std::vector<comm::Communicator>& TaskContext::communicators() const noexcept
{
  return comms_;
}

inline std::int64_t TaskContext::task_id() const noexcept { return task_->task_id; }

inline LegateVariantCode TaskContext::variant_kind() const noexcept { return variant_kind_; }

inline bool TaskContext::is_single_task() const { return !task_->is_index_space; }

inline bool TaskContext::can_raise_exception() const { return can_raise_exception_; }

inline bool TaskContext::can_elide_device_ctx_sync() const { return can_elide_device_ctx_sync_; }

inline const DomainPoint& TaskContext::get_task_index() const { return task_->index_point; }

inline const Domain& TaskContext::get_launch_domain() const { return task_->index_domain; }

inline void TaskContext::set_exception(ReturnedException what) { excn_ = std::move(what); }

inline std::optional<ReturnedException>& TaskContext::get_exception() noexcept { return excn_; }

inline const mapping::detail::Machine& TaskContext::machine() const { return machine_; }

inline std::string_view TaskContext::get_provenance() const
{
  return task_->get_provenance_string();
}

}  // namespace legate::detail
