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

#include "core/runtime/provenance_manager.h"

#include <assert.h>
#include <stdexcept>

namespace legate {

static const std::string BOTTOM = "";

ProvenanceManager::ProvenanceManager() { provenance_.push_back(BOTTOM); }

const std::string& ProvenanceManager::get_provenance()
{
#ifdef DEBUG_LEGATE
  assert(provenance_.size() > 0);
#endif
  return provenance_.back();
}

void ProvenanceManager::set_provenance(const std::string& p)
{
#ifdef DEBUG_LEGATE
  assert(provenance_.size() > 0);
#endif
  provenance_.back() = p;
}

void ProvenanceManager::reset_provenance()
{
#ifdef DEBUG_LEGATE
  assert(provenance_.size() > 0);
#endif
  provenance_.back() = BOTTOM;
}

void ProvenanceManager::push_provenance(const std::string& p) { provenance_.push_back(p); }

void ProvenanceManager::pop_provenance()
{
  if (provenance_.size() <= 1) {
    throw std::underflow_error("can't pop from an empty provenance stack");
  }
  provenance_.pop_back();
}

void ProvenanceManager::clear_all()
{
  provenance_.clear();
  provenance_.push_back(BOTTOM);
}

}  // namespace legate
