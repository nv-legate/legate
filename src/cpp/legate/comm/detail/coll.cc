/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/coll.h>

#include <legate_defines.h>

#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/local_network.h>
#include <legate/comm/detail/mpi_network.h>
#include <legate/comm/detail/ucc_network.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/macros.h>

#include <cstdlib>
#include <memory>
#include <utility>

namespace legate::detail::comm::coll {

// called from main thread
void init()
{
  // NOLINTBEGIN(bugprone-branch-clone)
  // The branch bodies may appear identical when certain compile-time flags are not defined,
  // but they serve different purposes based on LEGATE_USE_MPI and LEGATE_USE_UCX configuration.
  if (LEGATE_DEFINED(LEGATE_USE_MPI) && Runtime::get_runtime().config().need_network() &&
      !Runtime::get_runtime().config().disable_mpi()) {
#if LEGATE_DEFINED(LEGATE_USE_MPI)
    BackendNetwork::create_network(std::make_unique<detail::comm::coll::MPINetwork>());
#endif
  } else if (LEGATE_DEFINED(LEGATE_USE_UCX) && Runtime::get_runtime().config().need_network()) {
#if LEGATE_DEFINED(LEGATE_USE_UCX)
    BackendNetwork::create_network(std::make_unique<detail::comm::coll::UCCNetwork>());
#endif
  } else {
    BackendNetwork::create_network(std::make_unique<detail::comm::coll::LocalNetwork>());
  }
  // NOLINTEND(bugprone-branch-clone)
  // Make sure our nasty hack returned the right answer initially
  LEGATE_CHECK(BackendNetwork::get_network()->comm_type == BackendNetwork::guess_comm_type_());
}

void finalize()
{
  // Fully reset the BackendNetwork::get_network() pointer before letting the BackendNetwork object
  // get destroyed, so that any calls to LEGATE_ABORT within the destructor won't end up triggering
  // an abort while we're finalizing (MPI in particular doesn't like calls to MPI_Abort after MPI
  // has been finalized).
  if (BackendNetwork::has_network()) {
    std::ignore = std::exchange(BackendNetwork::get_network(), nullptr);
  }
}

void abort() noexcept
{
  if (BackendNetwork::has_network()) {
    BackendNetwork::get_network()->abort();
  }
  // If we are here, then either the backend network has not been initialized, or it didn't
  // have an abort mechanism. Either way, we abort normally now.
  std::abort();
}

}  // namespace legate::detail::comm::coll
