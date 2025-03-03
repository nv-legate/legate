/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/comm_cpu_factory.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/task_launcher.h>

namespace legate::detail::comm::cpu {

template <typename IT, typename IMT, typename FT>
Factory<IT, IMT, FT>::Factory(const detail::Library* core_library) : core_library_{core_library}
{
}

template <typename IT, typename IMT, typename FT>
bool Factory<IT, IMT, FT>::needs_barrier() const
{
  return false;
}

template <typename IT, typename IMT, typename FT>
bool Factory<IT, IMT, FT>::is_supported_target(mapping::TaskTarget /*target*/) const
{
  return true;
}

template <typename IT, typename IMT, typename FT>
Legion::FutureMap Factory<IT, IMT, FT>::initialize_(const mapping::detail::Machine& machine,
                                                    std::uint32_t num_tasks)
{
  const Domain launch_domain{
    Rect<1>{Point<1>{0}, Point<1>{static_cast<std::int64_t>(num_tasks) - 1}}};
  const auto tag = static_cast<Legion::MappingTagID>(machine.preferred_variant());
  // Generate a unique ID
  auto comm_id =
    Legion::Future::from_value<std::int32_t>(coll::BackendNetwork::get_network()->init_comm());
  // Find a mapping of all participants
  detail::TaskLauncher init_cpucoll_mapping_launcher{
    core_library_, machine, init_mapping_task_type::TASK_ID, tag};

  init_cpucoll_mapping_launcher.add_future(comm_id);
  // Setting this according to the return type on the task variant. Have to do this manually because
  // this launch is using the Legion task launcher directly.
  init_cpucoll_mapping_launcher.set_future_size(sizeof(int));

  const auto mapping = init_cpucoll_mapping_launcher.execute(launch_domain);

  // Then create communicators on participating processors
  detail::TaskLauncher init_cpucoll_launcher{core_library_, machine, init_task_type::TASK_ID, tag};

  init_cpucoll_launcher.add_future(comm_id);
  // Setting this according to the return type on the task variant. Have to do this manually because
  // this launch is using the Legion task launcher directly.
  init_cpucoll_launcher.set_future_size(sizeof(legate::comm::coll::CollComm));
  init_cpucoll_launcher.set_concurrent(true);

  const auto domain = mapping.get_future_map_domain();

  for (Domain::DomainPointIterator it{domain}; it; ++it) {
    init_cpucoll_launcher.add_future(mapping.get_future(*it));
  }
  return init_cpucoll_launcher.execute(launch_domain);
}

template <typename IT, typename IMT, typename FT>
void Factory<IT, IMT, FT>::finalize_(const mapping::detail::Machine& machine,
                                     std::uint32_t num_tasks,
                                     const Legion::FutureMap& communicator)
{
  const Domain launch_domain{
    Rect<1>{Point<1>{0}, Point<1>{static_cast<std::int64_t>(num_tasks) - 1}}};
  const auto tag = static_cast<Legion::MappingTagID>(machine.preferred_variant());
  detail::TaskLauncher launcher{core_library_, machine, finalize_task_type::TASK_ID, tag};

  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

}  // namespace legate::detail::comm::cpu
