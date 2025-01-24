/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/comm/detail/comm.h>

#include <legate_defines.h>

#include <legate/comm/detail/comm_cal.h>
#include <legate/comm/detail/comm_cpu.h>
#include <legate/comm/detail/comm_nccl.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/env_defaults.h>
#include <legate/utilities/macros.h>

namespace legate::detail::comm {

void register_tasks(Library* library)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    nccl::register_tasks(library);
  }
  if constexpr (LEGATE_DEFINED(LEGATE_USE_CAL)) {
    cal::register_tasks(library);
  }
  if (!LEGATE_DISABLE_MPI.get(LEGATE_DISABLE_MPI_DEFAULT, LEGATE_DISABLE_MPI_TEST)) {
    cpu::register_tasks(library);
  }
}

void register_builtin_communicator_factories(const Library* library)
{
  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    nccl::register_factory(library);
  }
  cpu::register_factory(library);
  if (LEGATE_DEFINED(LEGATE_USE_CAL)) {
    cal::register_factory(library);
  }
}

}  // namespace legate::detail::comm
