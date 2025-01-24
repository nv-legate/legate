/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/comm/detail/coll.h>

#include <legate_defines.h>

#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/local_network.h>
#include <legate/comm/detail/mpi_network.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/macros.h>

#include <cstdlib>
#include <memory>
#include <utility>

namespace legate::detail::comm::coll {

// called from main thread
void init()
{
  if (LEGATE_DEFINED(LEGATE_USE_NETWORK) &&
      detail::LEGATE_NEED_NETWORK.get(/* default_value */ false)) {
#if LEGATE_DEFINED(LEGATE_USE_NETWORK)
    BackendNetwork::create_network(std::make_unique<detail::comm::coll::MPINetwork>());
#endif
  } else {
    BackendNetwork::create_network(std::make_unique<detail::comm::coll::LocalNetwork>());
  }
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
