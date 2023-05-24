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

#include <string>
#include <vector>

namespace legate {

class ProvenanceManager {
 public:
  ProvenanceManager();

 public:
  const std::string& get_provenance();

  void set_provenance(const std::string& p);

  void reset_provenance();

  void push_provenance(const std::string& p);

  void pop_provenance();

  void clear_all();

 private:
  std::vector<std::string> provenance_;
};

}  // namespace legate
