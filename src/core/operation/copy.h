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

#include <memory>

#include "core/data/logical_store.h"
/**
 * @file
 * @brief Class definition for legate::Copy
 */

namespace legate::detail {
class Copy;
}  // namespace legate::detail

namespace legate {

class Copy {
 public:
  void add_input(LogicalStore store);
  void add_output(LogicalStore store);
  void add_reduction(LogicalStore store, Legion::ReductionOpID redop);
  void add_source_indirect(LogicalStore store);
  void add_target_indirect(LogicalStore store);

 public:
  void set_source_indirect_out_of_range(bool flag);
  void set_target_indirect_out_of_range(bool flag);

 public:
  Copy(const Copy&)            = delete;
  Copy(Copy&&)                 = default;
  Copy& operator=(const Copy&) = delete;
  Copy& operator=(Copy&&)      = default;

 public:
  ~Copy();

 private:
  friend class Runtime;
  Copy(std::unique_ptr<detail::Copy> impl);
  std::unique_ptr<detail::Copy> impl_;
};

}  // namespace legate
