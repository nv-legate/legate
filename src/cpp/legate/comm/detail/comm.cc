/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm.h>

#include <legate_defines.h>

#include <legate/comm/detail/comm_cal.h>
#include <legate/comm/detail/comm_cpu.h>
#include <legate/comm/detail/comm_nccl.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/macros.h>

namespace legate::detail::comm {

void register_tasks(Library& library)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_NCCL)) {
    nccl::register_tasks(library);
  }
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CAL)) {
    cal::register_tasks(library);
  }

  // Always register CPU communication tasks. The communicator factory selects
  // the appropriate backend (MPI, UCX, or the local collectives implementation)
  // internally, so even purely local builds still need these tasks registered
  // to satisfy add_cpu_communicator() callers.
  cpu::register_tasks(library);
}

void register_builtin_communicator_factories(const Library& library)
{
  if (LEGATE_DEFINED(LEGATE_USE_NCCL)) {
    nccl::register_factory(library);
  }
  cpu::register_factory(library);
  if (LEGATE_DEFINED(LEGATE_USE_CAL)) {
    cal::register_factory(library);
  }
}

}  // namespace legate::detail::comm
