/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/comm/coll.h"

#include "core/comm/detail/backend_network.h"
#include "core/comm/detail/local_network.h"
#include "core/comm/detail/logger.h"
#include "core/comm/detail/mpi_network.h"
#include "core/utilities/assert.h"
#include "core/utilities/env.h"
#include "core/utilities/macros.h"

#include "legate_defines.h"

namespace coll_detail = legate::detail::comm::coll;

namespace legate::comm::coll {

void collCommCreate(CollComm global_comm,
                    int global_comm_size,
                    int global_rank,
                    int unique_id,
                    const int* mapping_table)
{
  coll_detail::backend_network->comm_create(
    global_comm, global_comm_size, global_rank, unique_id, mapping_table);
}

void collCommDestroy(CollComm global_comm)
{
  coll_detail::backend_network->comm_destroy(global_comm);
}

void collAlltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   CollDataType type,
                   CollComm global_comm)
{
  // IN_PLACE
  if (sendbuf == recvbuf) {
    LEGATE_ABORT("Do not support inplace Alltoallv");
  }
  coll_detail::logger().debug() << "Alltoallv: global_rank " << global_comm->global_rank
                                << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                                << global_comm->unique_id << ", comm_size "
                                << global_comm->global_comm_size << ", mpi_comm_size "
                                << global_comm->mpi_comm_size << ' '
                                << global_comm->mpi_comm_size_actual << ", nb_threads "
                                << global_comm->nb_threads;

  coll_detail::backend_network->all_to_all_v(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
}

void collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  // IN_PLACE
  if (sendbuf == recvbuf) {
    LEGATE_ABORT("Do not support inplace Alltoall");
  }
  coll_detail::logger().debug() << "Alltoall: global_rank " << global_comm->global_rank
                                << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                                << global_comm->unique_id << ", comm_size "
                                << global_comm->global_comm_size << ", mpi_comm_size "
                                << global_comm->mpi_comm_size << ' '
                                << global_comm->mpi_comm_size_actual << ", nb_threads "
                                << global_comm->nb_threads;

  coll_detail::backend_network->all_to_all(sendbuf, recvbuf, count, type, global_comm);
}

void collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  coll_detail::logger().debug() << "Allgather: global_rank " << global_comm->global_rank
                                << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                                << global_comm->unique_id << ", comm_size "
                                << global_comm->global_comm_size << ", mpi_comm_size "
                                << global_comm->mpi_comm_size << ' '
                                << global_comm->mpi_comm_size_actual << ", nb_threads "
                                << global_comm->nb_threads;

  coll_detail::backend_network->all_gather(sendbuf, recvbuf, count, type, global_comm);
}

// called from main thread
void collInit(int argc, char* argv[])
{
  if (LEGATE_DEFINED(LEGATE_USE_NETWORK) && LEGATE_NEED_NETWORK.get().value_or(false)) {
#if LEGATE_DEFINED(LEGATE_USE_NETWORK)
    coll_detail::backend_network = std::make_unique<detail::comm::coll::MPINetwork>(argc, argv);
#endif
  } else {
    coll_detail::backend_network = std::make_unique<detail::comm::coll::LocalNetwork>(argc, argv);
  }
}

void collFinalize()
{
  // Fully reset the backend_network pointer before letting the BackendNetwork object get destroyed,
  // so that any calls to LEGATE_ABORT within the destructor won't end up triggering an abort while
  // we're finalizing (MPI in particular doesn't like calls to MPI_Abort after MPI has been
  // finalized).
  auto local_pointer = std::exchange(coll_detail::backend_network, nullptr);
}

void collAbort() noexcept
{
  if (coll_detail::backend_network) {
    coll_detail::backend_network->abort();
  }
}

int collInitComm() { return coll_detail::backend_network->init_comm(); }

}  // namespace legate::comm::coll
