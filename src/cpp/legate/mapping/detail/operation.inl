/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/operation.h>

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

inline std::size_t Task::future_size() const { return future_size_; }

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
