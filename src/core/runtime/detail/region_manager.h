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

#include "core/utilities/typedefs.h"

namespace legate::detail {

class Runtime;

class RegionManager {
 private:
  struct ManagerEntry {
    static constexpr Legion::FieldID FIELD_ID_BASE = 10000;
    static constexpr int32_t MAX_NUM_FIELDS = LEGION_MAX_FIELDS - LEGION_DEFAULT_LOCAL_FIELDS;

    ManagerEntry(const Legion::LogicalRegion& _region)
      : region(_region), next_field_id(FIELD_ID_BASE)
    {
    }
    bool has_space() const { return next_field_id - FIELD_ID_BASE < MAX_NUM_FIELDS; }
    Legion::FieldID get_next_field_id() { return next_field_id++; }

    void destroy(Runtime* runtime, bool unordered);

    Legion::LogicalRegion region;
    Legion::FieldID next_field_id;
  };

 public:
  RegionManager(Runtime* runtime, const Domain& shape);
  void destroy(bool unordered = false);

 private:
  const ManagerEntry& active_entry() const { return entries_.back(); }
  ManagerEntry& active_entry() { return entries_.back(); }
  void push_entry();

 public:
  bool has_space() const;
  std::pair<Legion::LogicalRegion, Legion::FieldID> allocate_field(size_t field_size);
  void import_region(const Legion::LogicalRegion& region);

 private:
  Runtime* runtime_;
  Domain shape_;
  std::vector<ManagerEntry> entries_{};
};

}  // namespace legate::detail
