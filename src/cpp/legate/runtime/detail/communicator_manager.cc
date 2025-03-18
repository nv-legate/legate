/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/communicator_manager.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

Legion::FutureMap CommunicatorFactory::find_or_create(const mapping::TaskTarget& target,
                                                      const mapping::ProcessorRange& range,
                                                      const Domain& launch_domain)
{
  if (launch_domain.dim == 1) {
    return find_or_create_(target, range, launch_domain.get_volume());
  }

  AliasKey key{launch_domain, target, range};
  auto finder = nd_aliases_.find(key);
  if (finder != nd_aliases_.end()) {
    return finder->second;
  }

  auto communicator = find_or_create_(target, range, launch_domain.get_volume());
  communicator      = transform_(communicator, launch_domain);
  nd_aliases_.insert({std::move(key), communicator});
  return communicator;
}

void CommunicatorFactory::destroy()
{
  for (auto&& [key, communicator] : communicators_) {
    finalize_(key.get_machine(), key.desc, communicator);
  }
  communicators_.clear();
  nd_aliases_.clear();
}

Legion::FutureMap CommunicatorFactory::find_or_create_(const mapping::TaskTarget& target,
                                                       const mapping::ProcessorRange& range,
                                                       std::uint32_t num_tasks)
{
  CommKey key{num_tasks, target, range};
  auto finder = communicators_.find(key);
  if (finder != communicators_.end()) {
    return finder->second;
  }

  auto communicator = initialize_(key.get_machine(), num_tasks);
  communicators_.insert({std::move(key), communicator});
  return communicator;
}

Legion::FutureMap CommunicatorFactory::transform_(const Legion::FutureMap& communicator,
                                                  const Domain& launch_domain)
{
  return Runtime::get_runtime()->delinearize_future_map(communicator, launch_domain);
}

// ==========================================================================================

std::optional<CommunicatorFactory*> CommunicatorManager::find_factory_(std::string_view name) const
{
  using pair_type = std::pair<std::string, std::unique_ptr<CommunicatorFactory>>;
  const auto it   = std::find_if(factories_.begin(), factories_.end(), [&](const pair_type& pair) {
    return pair.first == name;
  });

  return it == factories_.end() ? std::nullopt : std::make_optional(it->second.get());
}

CommunicatorFactory* CommunicatorManager::find_factory(std::string_view name) const
{
  if (const auto f = find_factory_(name); f.has_value()) {
    return *f;
  }
  throw TracedException<std::runtime_error>{
    fmt::format("No factory available for communicator '{}'", name)};
}

void CommunicatorManager::register_factory(std::string name,
                                           std::unique_ptr<CommunicatorFactory> factory)
{
  if (const auto f = find_factory_(name); f.has_value()) {
    throw TracedException<std::logic_error>{
      fmt::format("Factory '{}' already registered: {}", name, fmt::ptr(*f))};
  }
  factories_.emplace_back(std::move(name), std::move(factory));
}

void CommunicatorManager::destroy()
{
  for (auto it = factories_.rbegin(); it != factories_.rend(); ++it) {
    it->second->destroy();
    // ensure reverse task order due to dependencies (CAL->CPU)
    // without fence CPU teardown might supersede CAL
    Runtime::get_runtime()->issue_execution_fence();
  }
  factories_.clear();
}

}  // namespace legate::detail
