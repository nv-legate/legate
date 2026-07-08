/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm.h>

#include <legate_defines.h>

#include <legate/comm/detail/comm_cpu.h>
#include <legate/comm/detail/comm_nccl.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/macros.h>

namespace legate::detail::comm {

void register_builtin_communicator_factories(const Library& library)
{
  if (LEGATE_DEFINED(LEGATE_USE_NCCL)) {
    nccl::register_factory(library);
  }
  cpu::register_factory(library);
}

}  // namespace legate::detail::comm
