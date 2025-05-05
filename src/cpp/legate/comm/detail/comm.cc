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
#include <legate/runtime/detail/config.h>
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
  if (!Config::get_config().disable_mpi()) {
    cpu::register_tasks(library);
  }
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
