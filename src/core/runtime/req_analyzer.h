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

#include "legate.h"

namespace legate {

class RegionReq;

class RequirementAnalyzer {
 public:
  ~RequirementAnalyzer();

 public:
  void insert(RegionReq* req, Legion::FieldID field_id);
  uint32_t get_requirement_index(RegionReq* req, Legion::FieldID field_id) const;

 public:
  void analyze_requirements();
  void populate_launcher(Legion::IndexTaskLauncher* task) const;
  void populate_launcher(Legion::TaskLauncher* task) const;

 private:
  std::map<RegionReq*, uint32_t> req_indices_;
  std::vector<std::pair<RegionReq*, std::vector<Legion::FieldID>>> requirements_;
};

}  // namespace legate
