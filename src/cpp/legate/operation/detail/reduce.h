/* Copyright 2023-2025 NVIDIA Corporation
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

#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/operation.h>
#include <legate/utilities/internal_shared_ptr.h>

namespace legate::detail {

class Library;

class Reduce final : public Operation {
 public:
  Reduce(const Library& library,
         InternalSharedPtr<LogicalStore> store,
         InternalSharedPtr<LogicalStore> out_store,
         LocalTaskID task_id,
         std::uint64_t unique_id,
         std::int32_t radix,
         std::int32_t priority,
         mapping::detail::Machine machine);

  void launch(Strategy*) override;
  void validate() override;
  void add_to_solver(ConstraintSolver& solver) override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] bool needs_flush() const override;

 private:
  std::int32_t radix_{};
  std::reference_wrapper<const Library> library_;
  LocalTaskID task_id_{};
  InternalSharedPtr<LogicalStore> input_{};
  InternalSharedPtr<LogicalStore> output_{};
  const Variable* input_part_{};
  const Variable* output_part_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/reduce.inl>
