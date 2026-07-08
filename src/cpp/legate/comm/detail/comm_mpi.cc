/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm_mpi.h>

#include <legate_defines.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/comm_cpu_factory.h>
#include <legate/comm/detail/mpi_interface.h>
#include <legate/task/task_config.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/macros.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

namespace legate::detail::comm::mpi {

std::unique_ptr<CommunicatorFactory> make_factory(const legate::detail::Library&)
{
  return nullptr;
}

}  // namespace legate::detail::comm::mpi
