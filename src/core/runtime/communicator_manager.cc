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

#include "core/runtime/communicator_manager.h"

#include "core/runtime/machine_manager.h"
#include "core/runtime/runtime.h"

namespace legate {

CommunicatorFactory::CommunicatorFactory() {}

CommunicatorFactory::~CommunicatorFactory() {}

Legion::FutureMap CommunicatorFactory::find_or_create(const mapping::TaskTarget& target,
                                                      const mapping::ProcessorRange& range,
                                                      const Domain& launch_domain)
{
  if (launch_domain.dim == 1) return find_or_create(target, range, launch_domain.get_volume());

  AliasKey key{launch_domain, target, range};
  auto finder = nd_aliases_.find(key);
  if (finder != nd_aliases_.end()) return finder->second;

  auto communicator = find_or_create(target, range, launch_domain.get_volume());
  communicator      = transform(communicator, launch_domain);
  nd_aliases_.insert({key, communicator});
  return communicator;
}

void CommunicatorFactory::destroy()
{
  for (auto& [key, communicator] : communicators_)
    finalize(key.get_machine(), key.desc, communicator);
}

Legion::FutureMap CommunicatorFactory::find_or_create(const mapping::TaskTarget& target,
                                                      const mapping::ProcessorRange& range,
                                                      uint32_t num_tasks)
{
  CommKey key{num_tasks, target, range};
  auto finder = communicators_.find(key);
  if (finder != communicators_.end()) return finder->second;

  auto communicator = initialize(key.get_machine(), num_tasks);
  communicators_.insert({key, communicator});
  return communicator;
}

Legion::FutureMap CommunicatorFactory::transform(const Legion::FutureMap& communicator,
                                                 const Domain& launch_domain)
{
  auto* runtime   = Runtime::get_runtime();
  auto new_domain = runtime->find_or_create_index_space(launch_domain);
  return runtime->delinearize_future_map(communicator, new_domain);
}

CommunicatorFactory* CommunicatorManager::find_factory(const std::string& name)
{
  return factories_.at(name).get();
}

void CommunicatorManager::register_factory(const std::string& name,
                                           std::unique_ptr<CommunicatorFactory> factory)
{
  factories_.insert({name, std::move(factory)});
}

void CommunicatorManager::destroy()
{
  for (auto& [_, factory] : factories_) factory->destroy();
}

}  // namespace legate
