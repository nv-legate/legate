/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/coll.h>

#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/logger.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/traced_exception.h>

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
  if (sendbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid sendbuf: nullptr"};
  }
  if (recvbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid recvbuf: nullptr"};
  }
  if (sendcounts == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid sendcounts: nullptr"};
  }
  if (sdispls == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid sdispls: nullptr"};
  }
  if (recvcounts == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid recvcounts: nullptr"};
  }
  if (rdispls == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid rdispls: nullptr"};
  }
  // IN_PLACE is not supported
  if (sendbuf == recvbuf) {
    throw legate::detail::TracedException<std::invalid_argument>{
      "Inplace Alltoallv not yet supported"};
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
  if (sendbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid sendbuf: nullptr"};
  }
  if (recvbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid recvbuf: nullptr"};
  }
  if (count <= 0) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid count: <= 0"};
  }
  // IN_PLACE is not supported
  if (sendbuf == recvbuf) {
    throw legate::detail::TracedException<std::invalid_argument>{
      "Inplace Alltoall not yet supported"};
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
  if (sendbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid sendbuf: nullptr"};
  }
  if (recvbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid recvbuf: nullptr"};
  }
  if (count <= 0) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid count: <= 0"};
  }

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

void collAllreduce(const void* sendbuf,
                   void* recvbuf,
                   int count,
                   CollDataType type,
                   ReductionOpKind op,
                   CollComm global_comm)
{
  if (sendbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid sendbuf: nullptr"};
  }
  if (recvbuf == nullptr) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid recvbuf: nullptr"};
  }
  if (count <= 0) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid count: <= 0"};
  }

  switch (op) {
    case legate::ReductionOpKind::ADD: [[fallthrough]];
    case legate::ReductionOpKind::MUL: [[fallthrough]];
    case legate::ReductionOpKind::MAX: [[fallthrough]];
    case legate::ReductionOpKind::MIN: break;
    case legate::ReductionOpKind::OR: [[fallthrough]];
    case legate::ReductionOpKind::XOR: [[fallthrough]];
    case legate::ReductionOpKind::AND:
      if (type == legate::comm::coll::CollDataType::CollFloat ||
          type == legate::comm::coll::CollDataType::CollDouble) {
        throw legate::detail::TracedException<std::invalid_argument>{
          "all_reduce does not support float or double reduction with bitwise operations"};
      }
      break;
  }

  coll_detail::logger().debug() << "Allreduce: global_rank " << global_comm->global_rank
                                << ", mpi_rank " << global_comm->mpi_rank << ", unique_id "
                                << global_comm->unique_id << ", comm_size "
                                << global_comm->global_comm_size << ", mpi_comm_size "
                                << global_comm->mpi_comm_size << ' '
                                << global_comm->mpi_comm_size_actual << ", nb_threads "
                                << global_comm->nb_threads;

  coll_detail::BackendNetwork::get_network()->all_reduce(
    sendbuf, recvbuf, count, type, op, global_comm);
}

}  // namespace legate::comm::coll
