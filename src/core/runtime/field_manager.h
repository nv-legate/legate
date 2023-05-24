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

#include "core/utilities/typedefs.h"

namespace legate {

class LogicalRegionField;
class Runtime;

class FieldManager {
 public:
  FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size);

 public:
  std::shared_ptr<LogicalRegionField> allocate_field();
  std::shared_ptr<LogicalRegionField> import_field(const Legion::LogicalRegion& region,
                                                   Legion::FieldID field_id);

 private:
  Runtime* runtime_;
  Domain shape_;
  uint32_t field_size_;

 private:
  using FreeField = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::deque<FreeField> free_fields_;
};

}  // namespace legate
