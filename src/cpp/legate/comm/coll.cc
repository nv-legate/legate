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

#include "legate/comm/coll.h"

#include "legate/comm/detail/backend_network.h"
#include "legate/comm/detail/logger.h"
#include "legate/utilities/abort.h"

namespace coll_detail = legate::detail::comm::coll;

namespace legate::comm::coll {

void collCommCreate(CollComm global_comm,
                    int global_comm_size,
                    int global_rank,
                    int unique_id,
                    const int* mapping_table)
{
  coll_detail::BackendNetwork::get_network()->comm_create(
    global_comm, global_comm_size, global_rank, unique_id, mapping_table);
}

void collCommDestroy(CollComm global_comm)
{
  coll_detail::BackendNetwork::get_network()->comm_destroy(global_comm);
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

  coll_detail::BackendNetwork::get_network()->all_to_all_v(
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

  coll_detail::BackendNetwork::get_network()->all_to_all(
    sendbuf, recvbuf, count, type, global_comm);
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

  coll_detail::BackendNetwork::get_network()->all_gather(
    sendbuf, recvbuf, count, type, global_comm);
}

}  // namespace legate::comm::coll
