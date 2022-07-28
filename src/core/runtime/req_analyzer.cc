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

#include "core/runtime/req_analyzer.h"
#include "core/runtime/launcher.h"

namespace legate {

RequirementAnalyzer::~RequirementAnalyzer()
{
  for (auto& pair : requirements_) delete pair.first;
}

void RequirementAnalyzer::insert(RegionReq* req, Legion::FieldID field_id)
{
  uint32_t req_idx = static_cast<uint32_t>(requirements_.size());
  requirements_.push_back(std::make_pair(req, std::vector<Legion::FieldID>({field_id})));
  req_indices_[req] = req_idx;
}

uint32_t RequirementAnalyzer::get_requirement_index(RegionReq* req, Legion::FieldID field_id) const
{
  auto finder = req_indices_.find(req);
  assert(finder != req_indices_.end());
  return finder->second;
}

void RequirementAnalyzer::analyze_requirements() {}

void RequirementAnalyzer::populate_launcher(Legion::IndexTaskLauncher* task) const
{
  for (auto& pair : requirements_) {
    auto& req    = pair.first;
    auto& fields = pair.second;
    req->proj->populate_launcher(task, *req, fields);
  }
}

void RequirementAnalyzer::populate_launcher(Legion::TaskLauncher* task) const
{
  for (auto& pair : requirements_) {
    auto& req    = pair.first;
    auto& fields = pair.second;
    req->proj->populate_launcher(task, *req, fields);
  }
}

}  // namespace legate
