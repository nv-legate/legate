/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm_cpu.h>

#include <legate_defines.h>

#include <legate/comm/coll_comm.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/comm_local.h>
#include <legate/comm/detail/comm_mpi.h>
#include <legate/comm/detail/comm_ucc.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/library.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>

#include <stdexcept>

namespace legate::detail::comm::cpu {

void register_factory(const detail::Library& library)
{
  auto factory = [&]() -> std::unique_ptr<CommunicatorFactory> {
    using legate::comm::coll::CollCommType;

    switch (coll::BackendNetwork::guess_comm_type_()) {
      case CollCommType::CollMPI:
        if constexpr (LEGATE_DEFINED(LEGATE_USE_MPI)) {
          return mpi::make_factory(library);
        }
        throw legate::detail::TracedException<std::runtime_error>{
          "cannot create MPI factory, legate was not configured with MPI support"};
      case CollCommType::CollLocal: return local::make_factory(library);
      case CollCommType::CollUCC:
        if constexpr (LEGATE_DEFINED(LEGATE_USE_UCX)) {
          return legate::detail::comm::ucc::make_factory(library);
        }
        throw legate::detail::TracedException<std::runtime_error>{
          "cannot create UCC factory, legate was not configured with UCC support"};
    }
    LEGATE_UNREACHABLE();
    return nullptr;
  }();

  detail::Runtime::get_runtime().communicator_manager().register_factory("cpu", std::move(factory));
}

}  // namespace legate::detail::comm::cpu
