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

#include "mpi_wrapper_types.h"

#include <mpi.h>

#ifndef LEGATE_WEAK
#define LEGATE_WEAK
#endif

#ifdef __cplusplus
#define LEGATE_EXTERN extern "C"
#else
// extern technically isn't required in C, but it cannot hurt
#define LEGATE_EXTERN extern
#endif

#if defined(__cplusplus) && (__cplusplus >= 201103L)  // C++11
static_assert(sizeof(MPI_Status) <= LEGATE_MPI_STATUS_THUNK_SIZE,
              "Size of thunk too small to hold MPI_Status");
#elif defined(__STDC__) && defined(__STDC_VERSION__) && (__STDC__ == 1) && \
  (__STDC_VERSION__ >= 201112L)  // C11

#include <assert.h>  // technically no longer needed since C23

static_assert(sizeof(MPI_Status) <= LEGATE_MPI_STATUS_THUNK_SIZE,
              "Size of thunk too small to hold MPI_Status");
#endif

// NOLINTBEGIN
LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Comm legate_mpi_comm_world(void)
{
  return (Legate_MPI_Comm)MPI_COMM_WORLD;
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_thread_multiple(void) { return MPI_THREAD_MULTIPLE; }

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_tag_ub(void) { return MPI_TAG_UB; }

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_congruent(void) { return MPI_CONGRUENT; }

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_success(void) { return MPI_SUCCESS; }

// ==========================================================================================

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_int8_t(void)
{
  return (Legate_MPI_Datatype)MPI_INT8_T;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_uint8_t(void)
{
  return (Legate_MPI_Datatype)MPI_UINT8_T;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_char(void)
{
  return (Legate_MPI_Datatype)MPI_CHAR;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_byte(void)
{
  return (Legate_MPI_Datatype)MPI_BYTE;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_int(void)
{
  return (Legate_MPI_Datatype)MPI_INT;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_int32_t(void)
{
  return (Legate_MPI_Datatype)MPI_INT32_T;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_uint32_t(void)
{
  return (Legate_MPI_Datatype)MPI_UINT32_T;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_int64_t(void)
{
  return (Legate_MPI_Datatype)MPI_INT64_T;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_uint64_t(void)
{
  return (Legate_MPI_Datatype)MPI_UINT64_T;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_float(void)
{
  return (Legate_MPI_Datatype)MPI_FLOAT;
}

LEGATE_EXTERN LEGATE_WEAK Legate_MPI_Datatype legate_mpi_double(void)
{
  return (Legate_MPI_Datatype)MPI_DOUBLE;
}

// ==========================================================================================

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_init(int* argc, char*** argv)
{
  return MPI_Init(argc, argv);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_init_thread(int* argc,
                                                     char*** argv,
                                                     int required,
                                                     int* provided)
{
  return MPI_Init_thread(argc, argv, required, provided);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_finalize(void) { return MPI_Finalize(); }

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_abort(Legate_MPI_Comm comm, int error_code)
{
  return MPI_Abort((MPI_Comm)comm, error_code);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_initialized(int* init) { return MPI_Initialized(init); }

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_finalized(int* finalized)
{
  return MPI_Finalized(finalized);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_comm_dup(Legate_MPI_Comm comm, Legate_MPI_Comm* dup)
{
  return MPI_Comm_dup((MPI_Comm)comm, (MPI_Comm*)dup);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_comm_rank(Legate_MPI_Comm comm, int* rank)
{
  return MPI_Comm_rank((MPI_Comm)comm, rank);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_comm_size(Legate_MPI_Comm comm, int* size)
{
  return MPI_Comm_size((MPI_Comm)comm, size);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_comm_compare(Legate_MPI_Comm comm1,
                                                      Legate_MPI_Comm comm2,
                                                      int* result)
{
  return MPI_Comm_compare((MPI_Comm)comm1, (MPI_Comm)comm2, result);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_comm_get_attr(Legate_MPI_Comm comm,
                                                       int comm_keyval,
                                                       void* attribute_val,
                                                       int* flag)
{
  return MPI_Comm_get_attr((MPI_Comm)comm, comm_keyval, attribute_val, flag);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_comm_free(Legate_MPI_Comm* comm)
{
  return MPI_Comm_free((MPI_Comm*)comm);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_type_get_extent(Legate_MPI_Datatype type,
                                                         Legate_MPI_Aint* lb,
                                                         Legate_MPI_Aint* extent)
{
  return MPI_Type_get_extent((MPI_Datatype)type, (MPI_Aint*)lb, (MPI_Aint*)extent);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_query_thread(int* provided)
{
  return MPI_Query_thread(provided);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_bcast(
  void* buffer, int count, Legate_MPI_Datatype datatype, int root, Legate_MPI_Comm comm)
{
  return MPI_Bcast(buffer, count, (MPI_Datatype)datatype, root, (MPI_Comm)comm);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_send(
  const void* buf, int count, Legate_MPI_Datatype datatype, int dest, int tag, Legate_MPI_Comm comm)
{
  return MPI_Send(buf, count, (MPI_Datatype)datatype, dest, tag, (MPI_Comm)comm);
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_recv(void* buf,
                                              int count,
                                              Legate_MPI_Datatype datatype,
                                              int source,
                                              int tag,
                                              Legate_MPI_Comm comm,
                                              Legate_MPI_Status* status)
{
  MPI_Status* real_status = status ? (MPI_Status*)&status->original_private_ : MPI_STATUS_IGNORE;
  int ret = MPI_Recv(buf, count, (MPI_Datatype)datatype, source, tag, (MPI_Comm)comm, real_status);

  if (status) {
    status->MPI_SOURCE = real_status->MPI_SOURCE;
    status->MPI_TAG    = real_status->MPI_TAG;
    status->MPI_ERROR  = real_status->MPI_ERROR;
  }
  return ret;
}

LEGATE_EXTERN LEGATE_WEAK int legate_mpi_sendrecv(const void* sendbuf,
                                                  int sendcount,
                                                  Legate_MPI_Datatype sendtype,
                                                  int dest,
                                                  int sendtag,
                                                  void* recvbuf,
                                                  int recvcount,
                                                  Legate_MPI_Datatype recvtype,
                                                  int source,
                                                  int recvtag,
                                                  Legate_MPI_Comm comm,
                                                  Legate_MPI_Status* status)
{
  MPI_Status* real_status = status ? (MPI_Status*)&status->original_private_ : MPI_STATUS_IGNORE;
  int ret                 = MPI_Sendrecv(sendbuf,
                         sendcount,
                         (MPI_Datatype)sendtype,
                         dest,
                         sendtag,
                         recvbuf,
                         recvcount,
                         (MPI_Datatype)recvtype,
                         source,
                         recvtag,
                         (MPI_Comm)comm,
                         real_status);

  if (status) {
    status->MPI_SOURCE = real_status->MPI_SOURCE;
    status->MPI_TAG    = real_status->MPI_TAG;
    status->MPI_ERROR  = real_status->MPI_ERROR;
  }
  return ret;
}
// NOLINTEND
