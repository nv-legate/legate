/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/local_machine_selector.h>

#include <realm/network.h>

#include <unistd.h>

namespace legate::mapping::detail {

const LocalMachine& LocalMachineSelector::get_local() const
{
  return local_machine_cache_.try_emplace(Realm::Network::my_node_id).first->second;
}

const LocalMachine& LocalMachineSelector::get_local_to(Processor p) const
{
  return local_machine_cache_.try_emplace(p.address_space(), p).first->second;
}

const LocalMachine& LocalMachineSelector::get_local_to(Memory m) const
{
  return local_machine_cache_.try_emplace(m.address_space(), m).first->second;
}

}  // namespace legate::mapping::detail
