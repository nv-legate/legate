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

#include "legate/mapping/detail/operation.h"

namespace legate::mapping::detail {

inline Mappable::Mappable(const Legion::Mappable* mappable)
  : Mappable{private_tag{}, MapperDataDeserializer{mappable}}
{
}

inline const mapping::detail::Machine& Mappable::machine() const { return machine_; }

inline std::uint32_t Mappable::sharding_id() const { return sharding_id_; }

inline std::int32_t Mappable::priority() const { return priority_; }

// ==========================================================================================

inline legate::detail::Library* Task::library() { return library_; }

inline const legate::detail::Library* Task::library() const { return library_; }

inline const std::vector<InternalSharedPtr<Array>>& Task::inputs() const { return inputs_; }

inline const std::vector<InternalSharedPtr<Array>>& Task::outputs() const { return outputs_; }

inline const std::vector<InternalSharedPtr<Array>>& Task::reductions() const { return reductions_; }

inline const std::vector<InternalSharedPtr<legate::detail::Scalar>>& Task::scalars() const
{
  return scalars_;
}

inline bool Task::is_single_task() const { return !task_->is_index_space; }

inline const DomainPoint& Task::point() const { return task_->index_point; }

inline const Domain& Task::get_launch_domain() const { return task_->index_domain; }

inline bool Task::can_raise_exception() const { return can_raise_exception_; }

inline const Legion::Task* Task::legion_task() const { return task_; }

inline TaskTarget Task::target() const { return machine().preferred_target(); }

// ==========================================================================================

inline const std::vector<Store>& Copy::inputs() const { return inputs_; }

inline const std::vector<Store>& Copy::outputs() const { return outputs_; }

inline const std::vector<Store>& Copy::input_indirections() const { return input_indirections_; }

inline const std::vector<Store>& Copy::output_indirections() const { return output_indirections_; }

inline const DomainPoint& Copy::point() const { return copy_->index_point; }

}  // namespace legate::mapping::detail
