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

#include "coll.h"

#include "core/utilities/detail/strtoll.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

namespace legate::comm::coll {

namespace detail {

Logger& log_coll()
{
  static Logger log{"coll"};

  return log;
}

}  // namespace detail

BackendNetwork* backend_network = nullptr;

// functions start here
int collCommCreate(CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table)
{
  return backend_network->comm_create(
    global_comm, global_comm_size, global_rank, unique_id, mapping_table);
}

int collCommDestroy(CollComm global_comm) { return backend_network->comm_destroy(global_comm); }

int collAlltoallv(const void* sendbuf,
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
    detail::log_coll().error("Do not support inplace Alltoallv");
    LEGATE_ABORT;
  }
  detail::log_coll().debug(
    "Alltoallv: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d, "
    "mpi_comm_size %d %d, nb_threads %d",
    global_comm->global_rank,
    global_comm->mpi_rank,
    global_comm->unique_id,
    global_comm->global_comm_size,
    global_comm->mpi_comm_size,
    global_comm->mpi_comm_size_actual,
    global_comm->nb_threads);
  return backend_network->alltoallv(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
}

int collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  // IN_PLACE
  if (sendbuf == recvbuf) {
    detail::log_coll().error("Do not support inplace Alltoall");
    LEGATE_ABORT;
  }
  detail::log_coll().debug(
    "Alltoall: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d, "
    "mpi_comm_size %d %d, nb_threads %d",
    global_comm->global_rank,
    global_comm->mpi_rank,
    global_comm->unique_id,
    global_comm->global_comm_size,
    global_comm->mpi_comm_size,
    global_comm->mpi_comm_size_actual,
    global_comm->nb_threads);
  return backend_network->alltoall(sendbuf, recvbuf, count, type, global_comm);
}

int collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  detail::log_coll().debug(
    "Allgather: global_rank %d, mpi_rank %d, unique_id %d, comm_size %d, "
    "mpi_comm_size %d %d, nb_threads %d",
    global_comm->global_rank,
    global_comm->mpi_rank,
    global_comm->unique_id,
    global_comm->global_comm_size,
    global_comm->mpi_comm_size,
    global_comm->mpi_comm_size_actual,
    global_comm->nb_threads);
  return backend_network->allgather(sendbuf, recvbuf, count, type, global_comm);
}

// called from main thread
int collInit(int argc, char* argv[])
{
  if (LegateDefined(LEGATE_USE_NETWORK)) {
    char* network    = getenv("LEGATE_NEED_NETWORK");
    int need_network = 0;
    if (network != nullptr) {
      need_network = legate::detail::safe_strtoll<int>(network);
    }
    if (need_network) {
#if LegateDefined(LEGATE_USE_NETWORK)
      backend_network = new MPINetwork{argc, argv};
#endif
    } else {
      backend_network = new LocalNetwork{argc, argv};
    }
  } else {
    backend_network = new LocalNetwork{argc, argv};
  }
  return CollSuccess;
}

int collFinalize()
{
  delete backend_network;
  return CollSuccess;
}

int collInitComm() { return backend_network->init_comm(); }

int BackendNetwork::collGetUniqueId(int* id)
{
  *id = current_unique_id;
  current_unique_id++;
  return CollSuccess;
}

void* BackendNetwork::allocateInplaceBuffer(const void* recvbuf, size_t size)
{
  void* sendbuf_tmp = malloc(size);
  assert(sendbuf_tmp != nullptr);
  memcpy(sendbuf_tmp, recvbuf, size);
  return sendbuf_tmp;
}

}  // namespace legate::comm::coll

extern "C" {

int legate_cpucoll_finalize(void)
{
  // REVIEW: this should be returned?
  return legate::comm::coll::collFinalize();
}

int legate_cpucoll_initcomm(void) { return legate::comm::coll::collInitComm(); }
}
