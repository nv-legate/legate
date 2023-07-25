/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/comm/comm.h"
#ifdef LEGATE_USE_CUDA
#include "core/comm/comm_nccl.h"
#endif
#include "core/comm/comm_cpu.h"
#include "core/runtime/runtime.h"
#include "env_defaults.h"

namespace legate::comm {

void register_tasks(Legion::Runtime* runtime, const detail::Library* library)
{
#ifdef LEGATE_USE_CUDA
  nccl::register_tasks(runtime, library);
#endif
  bool disable_mpi =
    static_cast<bool>(extract_env("LEGATE_DISABLE_MPI", DISABLE_MPI_DEFAULT, DISABLE_MPI_TEST));
  if (!disable_mpi) { cpu::register_tasks(runtime, library); }
}

void register_builtin_communicator_factories(const detail::Library* library)
{
#ifdef LEGATE_USE_CUDA
  nccl::register_factory(library);
#endif
  cpu::register_factory(library);
}

}  // namespace legate::comm
