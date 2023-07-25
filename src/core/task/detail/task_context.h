/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "core/comm/communicator.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/detail/machine.h"
#include "core/task/detail/return.h"

namespace legate::detail {

class TaskContext {
 public:
  TaskContext(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions);

 public:
  std::vector<legate::Store>& inputs() { return inputs_; }
  std::vector<legate::Store>& outputs() { return outputs_; }
  std::vector<legate::Store>& reductions() { return reductions_; }
  std::vector<legate::Scalar>& scalars() { return scalars_; }
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
  std::vector<legate::Store> inputs_, outputs_, reductions_;
  std::vector<legate::Store> unbound_stores_;
  std::vector<legate::Store> scalar_stores_;
  std::vector<legate::Scalar> scalars_;
  std::vector<comm::Communicator> comms_;
  bool can_raise_exception_;
  mapping::detail::Machine machine_;
};

}  // namespace legate::detail
