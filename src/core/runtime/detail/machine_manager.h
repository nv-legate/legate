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

#include "core/mapping/detail/machine.h"

namespace legate::mapping::detail {
class Machine;
}  // namespace legate::mapping::detail

namespace legate::detail {

class MachineManager {
 public:
  const mapping::detail::Machine& get_machine() const;

  void push_machine(mapping::detail::Machine&& machine);

  void pop_machine();

 private:
  std::vector<legate::mapping::detail::Machine> machines_;
};

}  // namespace legate::detail
