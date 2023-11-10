/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/runtime/detail/communicator_manager.h"

#include "core/runtime/detail/runtime.h"

namespace legate::detail {

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
  nd_aliases_.insert({std::move(key), communicator});
  return communicator;
}

void CommunicatorFactory::destroy()
{
  for (auto& [key, communicator] : communicators_) {
    finalize(key.get_machine(), key.desc, communicator);
  }
  communicators_.clear();
  nd_aliases_.clear();
}

Legion::FutureMap CommunicatorFactory::find_or_create(const mapping::TaskTarget& target,
                                                      const mapping::ProcessorRange& range,
                                                      uint32_t num_tasks)
{
  CommKey key{num_tasks, target, range};
  auto finder = communicators_.find(key);
  if (finder != communicators_.end()) return finder->second;

  auto communicator = initialize(key.get_machine(), num_tasks);
  communicators_.insert({std::move(key), communicator});
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
  factories_.clear();
}

}  // namespace legate::detail
