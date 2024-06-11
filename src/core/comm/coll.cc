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

#include "core/comm/backend_network.h"
#include "core/comm/local_network.h"
#include "core/comm/mpi_network.h"
#include "core/utilities/env.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace legate::comm::coll {

namespace detail {

Logger& log_coll()
{
  static Logger log{"coll"};

  return log;
}

}  // namespace detail

CollStatus collCommCreate(CollComm global_comm,
                          int global_comm_size,
                          int global_rank,
                          int unique_id,
                          const int* mapping_table)
{
  return backend_network->comm_create(
    global_comm, global_comm_size, global_rank, unique_id, mapping_table);
}

CollStatus collCommDestroy(CollComm global_comm)
{
  return backend_network->comm_destroy(global_comm);
}

CollStatus collAlltoallv(const void* sendbuf,
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
  detail::log_coll().debug() << "Alltoallv: global_rank " << global_comm->global_rank
                             << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                             << global_comm->unique_id << ", comm_size "
                             << global_comm->global_comm_size << ", mpi_comm_size "
                             << global_comm->mpi_comm_size << ' '
                             << global_comm->mpi_comm_size_actual << ", nb_threads "
                             << global_comm->nb_threads;

  return backend_network->all_to_all_v(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
}

CollStatus collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  // IN_PLACE
  if (sendbuf == recvbuf) {
    LEGATE_ABORT("Do not support inplace Alltoall");
  }
  detail::log_coll().debug() << "Alltoall: global_rank " << global_comm->global_rank
                             << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                             << global_comm->unique_id << ", comm_size "
                             << global_comm->global_comm_size << ", mpi_comm_size "
                             << global_comm->mpi_comm_size << ' '
                             << global_comm->mpi_comm_size_actual << ", nb_threads "
                             << global_comm->nb_threads;

  return backend_network->all_to_all(sendbuf, recvbuf, count, type, global_comm);
}

CollStatus collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  detail::log_coll().debug() << "Allgather: global_rank " << global_comm->global_rank
                             << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                             << global_comm->unique_id << ", comm_size "
                             << global_comm->global_comm_size << ", mpi_comm_size "
                             << global_comm->mpi_comm_size << ' '
                             << global_comm->mpi_comm_size_actual << ", nb_threads "
                             << global_comm->nb_threads;

  return backend_network->all_gather(sendbuf, recvbuf, count, type, global_comm);
}

// called from main thread
CollStatus collInit(int argc, char* argv[])
{
  if (LEGATE_DEFINED(LEGATE_USE_NETWORK) && LEGATE_NEED_NETWORK.get().value_or(false)) {
#if LEGATE_DEFINED(LEGATE_USE_NETWORK)
    backend_network = std::make_unique<MPINetwork>(argc, argv);
#endif
  } else {
    backend_network = std::make_unique<LocalNetwork>(argc, argv);
  }
  return CollSuccess;
}

CollStatus collFinalize()
{
  backend_network.reset();
  return CollSuccess;
}

void collAbort() noexcept
{
  if (backend_network) {
    backend_network->abort();
  }
}

int collInitComm() { return backend_network->init_comm(); }

void BackendNetwork::abort()
{
  // does nothing by default
}

CollStatus BackendNetwork::get_unique_id_(int* id)
{
  *id = current_unique_id_++;
  return CollSuccess;
}

void* BackendNetwork::allocate_inplace_buffer_(const void* recvbuf, std::size_t size)
{
  LEGATE_ASSERT(size);
  void* sendbuf_tmp = std::malloc(size);
  LEGATE_CHECK(sendbuf_tmp != nullptr);
  std::memcpy(sendbuf_tmp, recvbuf, size);
  return sendbuf_tmp;
}

void BackendNetwork::delete_inplace_buffer_(void* recvbuf, std::size_t) { std::free(recvbuf); }

}  // namespace legate::comm::coll

extern "C" {

int legate_cpucoll_finalize() { return legate::comm::coll::collFinalize(); }

int legate_cpucoll_initcomm() { return legate::comm::coll::collInitComm(); }

bool legate_has_cal() { return LEGATE_DEFINED(LEGATE_USE_CAL); }

}  //
