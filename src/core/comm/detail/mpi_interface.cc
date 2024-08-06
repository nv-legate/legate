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
#include "core/comm/detail/mpi_interface.h"

#include <legate_mpi_wrapper/mpi_wrapper.h>

namespace legate::detail::comm::mpi::detail {

/*static*/ MPIInterface::MPI_Comm MPIInterface::MPI_COMM_WORLD() { return legate_mpi_comm_world(); }

/*static*/ int MPIInterface::MPI_THREAD_MULTIPLE() { return legate_mpi_thread_multiple(); }

/*static*/ int MPIInterface::MPI_TAG_UB() { return legate_mpi_tag_ub(); }

/*static*/ int MPIInterface::MPI_CONGRUENT() { return legate_mpi_congruent(); }

/*static*/ int MPIInterface::MPI_SUCCESS() { return legate_mpi_success(); }

// ==========================================================================================

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT8_T() { return legate_mpi_int8_t(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_UINT8_T() { return legate_mpi_uint8_t(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_CHAR() { return legate_mpi_char(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_BYTE() { return legate_mpi_byte(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT() { return legate_mpi_int(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT32_T() { return legate_mpi_int32_t(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_UINT32_T() { return legate_mpi_uint32_t(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT64_T() { return legate_mpi_int64_t(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_UINT64_T() { return legate_mpi_uint64_t(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_FLOAT() { return legate_mpi_float(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_DOUBLE() { return legate_mpi_double(); }

// ==========================================================================================

/*static*/ int MPIInterface::mpi_init_thread(int* argc, char*** argv, int required, int* provided)
{
  return legate_mpi_init_thread(argc, argv, required, provided);
}

/*static*/ int MPIInterface::mpi_finalize() { return legate_mpi_finalize(); }

/*static*/ int MPIInterface::mpi_abort(MPIInterface::MPI_Comm comm, int error_code)
{
  return legate_mpi_abort(comm, error_code);
}

/*static*/ int MPIInterface::mpi_initialized(int* init) { return legate_mpi_initialized(init); }

/*static*/ int MPIInterface::mpi_finalized(int* finalized)
{
  return legate_mpi_finalized(finalized);
}

/*static*/ int MPIInterface::mpi_comm_dup(MPIInterface::MPI_Comm comm, MPIInterface::MPI_Comm* dup)
{
  return legate_mpi_comm_dup(comm, dup);
}

/*static*/ int MPIInterface::mpi_comm_rank(MPIInterface::MPI_Comm comm, int* rank)
{
  return legate_mpi_comm_rank(comm, rank);
}

/*static*/ int MPIInterface::mpi_comm_size(MPIInterface::MPI_Comm comm, int* size)
{
  return legate_mpi_comm_size(comm, size);
}

/*static*/ int MPIInterface::mpi_comm_compare(MPIInterface::MPI_Comm comm1,
                                              MPIInterface::MPI_Comm comm2,
                                              int* result)
{
  return legate_mpi_comm_compare(comm1, comm2, result);
}

/*static*/ int MPIInterface::mpi_comm_get_attr(MPIInterface::MPI_Comm comm,
                                               int comm_keyval,
                                               void* attribute_val,
                                               int* flag)
{
  return legate_mpi_comm_get_attr(comm, comm_keyval, attribute_val, flag);
}

/*static*/ int MPIInterface::mpi_comm_free(MPIInterface::MPI_Comm* comm)
{
  return legate_mpi_comm_free(comm);
}

/*static*/ int MPIInterface::mpi_type_get_extent(MPIInterface::MPI_Datatype type,
                                                 MPIInterface::MPI_Aint* lb,
                                                 MPIInterface::MPI_Aint* extent)
{
  return legate_mpi_type_get_extent(type, lb, extent);
}

/*static*/ int MPIInterface::mpi_query_thread(int* provided)
{
  return legate_mpi_query_thread(provided);
}

/*static*/ int MPIInterface::mpi_bcast(void* buffer,
                                       int count,
                                       MPIInterface::MPI_Datatype datatype,
                                       int root,
                                       MPIInterface::MPI_Comm comm)
{
  return legate_mpi_bcast(buffer, count, datatype, root, comm);
}

/*static*/ int MPIInterface::mpi_send(const void* buf,
                                      int count,
                                      MPIInterface::MPI_Datatype datatype,
                                      int dest,
                                      int tag,
                                      MPIInterface::MPI_Comm comm)
{
  return legate_mpi_send(buf, count, datatype, dest, tag, comm);
}

/*static*/ int MPIInterface::mpi_recv(void* buf,
                                      int count,
                                      MPIInterface::MPI_Datatype datatype,
                                      int source,
                                      int tag,
                                      MPIInterface::MPI_Comm comm,
                                      MPIInterface::MPI_Status* status)
{
  return legate_mpi_recv(buf, count, datatype, source, tag, comm, status);
}

/*static*/ int MPIInterface::mpi_sendrecv(const void* sendbuf,
                                          int sendcount,
                                          MPIInterface::MPI_Datatype sendtype,
                                          int dest,
                                          int sendtag,
                                          void* recvbuf,
                                          int recvcount,
                                          MPIInterface::MPI_Datatype recvtype,
                                          int source,
                                          int recvtag,
                                          MPIInterface::MPI_Comm comm,
                                          MPIInterface::MPI_Status* status)
{
  return legate_mpi_sendrecv(sendbuf,
                             sendcount,
                             sendtype,
                             dest,
                             sendtag,
                             recvbuf,
                             recvcount,
                             recvtype,
                             source,
                             recvtag,
                             comm,
                             status);
}

}  // namespace legate::detail::comm::mpi::detail
