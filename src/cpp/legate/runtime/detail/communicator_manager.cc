/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

  auto&& runtime = Runtime::get_runtime();
  // Without this fence, Legion might still reorder the finalization tasks if they don't have
  // obvious data dependencies.
  //
  // For example, a communicator can use another communicator inside, so the
  // finalize task doesn't need to pass both these communicators as arguments to the
  // task. So Legion thinks they are disjoint and schedules them independently.
  runtime.issue_execution_fence();
  // Must also flush the scheduling window so that our fence definitely makes it down to
  // Legion. This is needed because the communicator tasks bypass the runtime pipelines by
  // using `TaskLauncher`, which submits directly to Legion.
  //
  // This leads to the case where Legion sees the task submissions, but we don't, and so
  // `issue_execution_fence()` doesn't do anything because our queue is empty.
  runtime.flush_scheduling_window();
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
  return Runtime::get_runtime().delinearize_future_map(communicator, launch_domain);
}

// ==========================================================================================

std::optional<std::reference_wrapper<CommunicatorFactory>> CommunicatorManager::find_factory_(
  std::string_view name) const
{
  using pair_type = std::pair<std::string, std::unique_ptr<CommunicatorFactory>>;
  const auto it   = std::find_if(factories_.begin(), factories_.end(), [&](const pair_type& pair) {
    return pair.first == name;
  });

  if (it == factories_.end()) {
    return std::nullopt;
  }
  return *(it->second);
}

CommunicatorFactory& CommunicatorManager::find_factory(std::string_view name) const
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
      fmt::format("Factory '{}' already registered: {}", name, fmt::ptr(&f->get()))};
  }
  factories_.emplace_back(std::move(name), std::move(factory));
}

void CommunicatorManager::destroy()
{
  // Communicator factories should be destroyed in reverse order of creation, to ensure that
  // any dependencies (e.g. CAL depends on CPU) are destroyed in hierarchical order.
  for (auto it = factories_.rbegin(); it != factories_.rend(); ++it) {
    it->second->destroy();
  }
  factories_.clear();
}

}  // namespace legate::detail
