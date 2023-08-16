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

#include "core/comm/communicator.h"
#include "core/data/detail/array.h"
#include "core/data/scalar.h"
#include "core/mapping/detail/machine.h"
#include "core/task/detail/return.h"

namespace legate::detail {

class TaskContext {
 public:
  TaskContext(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions);

 public:
  std::vector<std::shared_ptr<Array>>& inputs() { return inputs_; }
  std::vector<std::shared_ptr<Array>>& outputs() { return outputs_; }
  std::vector<std::shared_ptr<Array>>& reductions() { return reductions_; }
  const std::vector<legate::Scalar>& scalars() { return scalars_; }
  std::vector<comm::Communicator>& communicators() { return comms_; }

 public:
  bool is_single_task() const { return !task_->is_index_space; }
  bool can_raise_exception() const { return can_raise_exception_; }
  DomainPoint get_task_index() const { return task_->index_point; }
  Domain get_launch_domain() const { return task_->index_domain; }

 public:
  const mapping::detail::Machine& machine() const { return machine_; }
  const std::string& get_provenance() const { return task_->get_provenance_string(); }

 public:
  /**
   * @brief Makes all of unbound output stores of this task empty
   */
  void make_all_unbound_stores_empty();
  ReturnValues pack_return_values() const;
  ReturnValues pack_return_values_with_exception(int32_t index,
                                                 const std::string& error_message) const;

 private:
  std::vector<ReturnValue> get_return_values() const;

 private:
  const Legion::Task* task_;
  const std::vector<Legion::PhysicalRegion>& regions_;

 private:
  std::vector<std::shared_ptr<Array>> inputs_, outputs_, reductions_;
  std::vector<std::shared_ptr<Store>> unbound_stores_;
  std::vector<std::shared_ptr<Store>> scalar_stores_;
  std::vector<legate::Scalar> scalars_;
  std::vector<comm::Communicator> comms_;
  bool can_raise_exception_;
  mapping::detail::Machine machine_;
};

}  // namespace legate::detail
