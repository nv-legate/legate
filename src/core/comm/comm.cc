/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate_defines.h"
//
#include "core/comm/comm.h"
#include "core/comm/comm_cpu.h"
#include "core/comm/comm_nccl.h"
#include "core/runtime/runtime.h"
#include "env_defaults.h"

namespace legate::comm {

void register_tasks(Legion::Runtime* runtime, const detail::Library* library)
{
  if (LegateDefined(LEGATE_USE_CUDA)) { nccl::register_tasks(runtime, library); }
  bool disable_mpi =
    static_cast<bool>(extract_env("LEGATE_DISABLE_MPI", DISABLE_MPI_DEFAULT, DISABLE_MPI_TEST));
  if (!disable_mpi) { cpu::register_tasks(runtime, library); }
}

void register_builtin_communicator_factories(const detail::Library* library)
{
  if (LegateDefined(LEGATE_USE_CUDA)) { nccl::register_factory(library); }
  cpu::register_factory(library);
}

}  // namespace legate::comm
