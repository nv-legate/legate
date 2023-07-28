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

#include "core/operation/detail/operation.h"

#include "core/mapping/detail/machine.h"

namespace legate::detail {

class Reduce : public Operation {
 private:
  friend class Runtime;
  Reduce(const Library* library,
         std::shared_ptr<LogicalStore> store,
         std::shared_ptr<LogicalStore> out_store,
         int64_t task_id,
         int64_t unique_id,
         int64_t radix,
         mapping::detail::Machine&& machine);

 public:
  void launch(Strategy*) override;

 public:
  void validate() override;
  void add_to_solver(ConstraintSolver& solver) override;

 public:
  std::string to_string() const override;

 private:
  int64_t radix_;
  const Library* library_;
  int64_t task_id_;
  std::shared_ptr<LogicalStore> input_;
  std::shared_ptr<LogicalStore> output_;
  const Variable* input_part_;
  const Variable* output_part_;
};

}  // namespace legate::detail
