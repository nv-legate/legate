/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/comm_cpu_factory.h>
#include <legate/mapping/detail/machine.h>
#include <legate/tuning/parallel_policy.h>

namespace legate::detail::comm::cpu {

template <typename IT, typename IMT, typename FT>
Factory<IT, IMT, FT>::Factory(const detail::Library& core_library) : core_library_{&core_library}
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
Legion::FutureMap Factory<IT, IMT, FT>::initialize_(const mapping::detail::Machine&, std::uint32_t)
{
  // TODO(wonchanl): Need a v2 implementation
  return {};
}

template <typename IT, typename IMT, typename FT>
void Factory<IT, IMT, FT>::finalize_(const mapping::detail::Machine&,
                                     std::uint32_t,
                                     const Legion::FutureMap&)
{
  // TODO(wonchanl): Need a v2 implementation
}

}  // namespace legate::detail::comm::cpu
