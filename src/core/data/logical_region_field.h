/* Copyright 2022 NVIDIA Corporation
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
#include <unordered_map>

#include "legion.h"

#include "core/data/shape.h"

namespace legate {

class Partition;
class Tiling;

class LogicalRegionField : public std::enable_shared_from_this<LogicalRegionField> {
 public:
  LogicalRegionField() {}
  LogicalRegionField(const Legion::LogicalRegion& lr,
                     Legion::FieldID fid,
                     std::shared_ptr<LogicalRegionField> parent = nullptr);

 public:
  LogicalRegionField(const LogicalRegionField& other)            = default;
  LogicalRegionField& operator=(const LogicalRegionField& other) = default;

 public:
  int32_t dim() const;
  const Legion::LogicalRegion& region() const { return lr_; }
  Legion::FieldID field_id() const { return fid_; }
  const LogicalRegionField& get_root() const;

 public:
  Legion::Domain domain() const;

 public:
  std::shared_ptr<LogicalRegionField> get_child(const Tiling* tiling,
                                                const Shape& color,
                                                bool complete);
  Legion::LogicalPartition get_legion_partition(const Partition* partition, bool complete);

 private:
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{-1U};
  std::shared_ptr<LogicalRegionField> parent_{nullptr};
};

}  // namespace legate
