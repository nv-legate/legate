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

#include <map>
#include <unordered_map>

#include "core/mapping/machine.h"
#include "core/utilities/typedefs.h"

namespace legate {

class CommunicatorFactory {
 protected:
  CommunicatorFactory();

 public:
  virtual ~CommunicatorFactory();

 public:
  Legion::FutureMap find_or_create(const mapping::TaskTarget& target,
                                   const mapping::ProcessorRange& range,
                                   const Domain& launch_domain);
  void destroy();

 protected:
  Legion::FutureMap find_or_create(const mapping::TaskTarget& target,
                                   const mapping::ProcessorRange& range,
                                   uint32_t num_tasks);
  Legion::FutureMap transform(const Legion::FutureMap& communicator, const Domain& launch_domain);

 public:
  virtual bool needs_barrier() const                                 = 0;
  virtual bool is_supported_target(mapping::TaskTarget target) const = 0;

 protected:
  virtual Legion::FutureMap initialize(const mapping::MachineDesc& machine, uint32_t num_tasks) = 0;
  virtual void finalize(const mapping::MachineDesc& machine,
                        uint32_t num_tasks,
                        const Legion::FutureMap& communicator)                                  = 0;

 private:
  template <class Desc>
  struct CacheKey {
    Desc desc;
    mapping::TaskTarget target;
    mapping::ProcessorRange range;
    mapping::MachineDesc get_machine() const { return mapping::MachineDesc({{target, range}}); }
    bool operator==(const CacheKey& other) const
    {
      return desc == other.desc && target == other.target && range == other.range;
    }
    bool operator<(const CacheKey& other) const
    {
      if (desc < other.desc)
        return true;
      else if (other.desc < desc)
        return false;
      if (target < other.target)
        return true;
      else if (target > other.target)
        return false;
      return range < other.range;
    }
  };
  using CommKey  = CacheKey<uint32_t>;
  using AliasKey = CacheKey<Domain>;
  std::map<CommKey, Legion::FutureMap> communicators_;
  std::map<AliasKey, Legion::FutureMap> nd_aliases_;
};

class CommunicatorManager {
 public:
  CommunicatorFactory* find_factory(const std::string& name);
  void register_factory(const std::string& name, std::unique_ptr<CommunicatorFactory> factory);

 public:
  void destroy();

 private:
  std::unordered_map<std::string, std::unique_ptr<CommunicatorFactory>> factories_;
};

}  // namespace legate
